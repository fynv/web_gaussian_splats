import { DataUtils } from "./engine/extras/DataUtils.js"
import { uintEncodedFloat, rgbaToInteger } from './Util.js';

import { ResetIndices } from "./reset_indices.js"
import { CalcKey } from "./calc_key.js"
import { KeyReduction } from "./reduce_key.js"
import { RadixSort, FrustumCull } from "./radix_sort.js"


const workgroup_size = 64;
const workgroup_size_2x = workgroup_size*2;


export class Splats
{
    constructor(splatBuffer)
    {
        this.splatBuffer = splatBuffer;
        this._initialize();
        this.r_aperture = 0.0;
        this.focal_z = 3.0;
    }

    set_r_aperture(r_aperture)
    {
        this.r_aperture = r_aperture;
        this.updateConstant();
    }

    set_focal_z(focal_z)
    {
        this.focal_z = focal_z;
        this.updateConstant();
    }

    _initialize()
    {
        const splatCount = this.splatBuffer.getSplatCount();

        this.covariances = new Float32Array(splatCount * 6);
        this.colors = new Uint8Array(splatCount * 4);
        this.centers = new Float32Array(splatCount * 3);
        this.splatBuffer.fillCovarianceArray(this.covariances);
        this.splatBuffer.fillCenterArray(this.centers);
        this.splatBuffer.fillColorArray(this.colors);

        const covariancesTextureSize = [4096, 1024];
        while (covariancesTextureSize[0] * covariancesTextureSize[1] * 2 < splatCount * 6) {
            covariancesTextureSize[1] *= 2;
        }

        const centersColorsTextureSize = [4096, 1024];
        while (centersColorsTextureSize[0] * centersColorsTextureSize[1] * 4 < splatCount * 4) {
            centersColorsTextureSize[1] *= 2;
        }

        const paddedCovariances = new Uint16Array(covariancesTextureSize[0] * covariancesTextureSize[1] * 2);
        for (let i = 0; i < this.covariances.length; i++) {
            paddedCovariances[i] = DataUtils.toHalfFloat(this.covariances[i]);
        }

        const paddedCenterColors = new Uint32Array(centersColorsTextureSize[0] * centersColorsTextureSize[1] * 4);
        for (let c = 0; c < splatCount; c++) {
            const colorsBase = c * 4;
            const centersBase = c * 3;
            const centerColorsBase = c * 4;
            paddedCenterColors[centerColorsBase] = rgbaToInteger(this.colors[colorsBase], this.colors[colorsBase + 1], this.colors[colorsBase + 2], this.colors[colorsBase + 3]);
            paddedCenterColors[centerColorsBase + 1] = uintEncodedFloat(this.centers[centersBase]);
            paddedCenterColors[centerColorsBase + 2] = uintEncodedFloat(this.centers[centersBase + 1]);
            paddedCenterColors[centerColorsBase + 3] = uintEncodedFloat(this.centers[centersBase + 2]);
        }

        const stageBuf_Covariances =  engine_ctx.createBuffer(paddedCovariances.buffer, GPUBufferUsage.COPY_SRC);
        const stageBuf_CenterColors =  engine_ctx.createBuffer(paddedCenterColors.buffer, GPUBufferUsage.COPY_SRC);

        this.covariancesTexture = engine_ctx.device.createTexture({            
            dimension: "2d",
            size: { width: covariancesTextureSize[0], height: covariancesTextureSize[1]},
            format: "rg16float",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
        });

        this.centersColorsTexture = engine_ctx.device.createTexture({            
            dimension: "2d",
            size: { width: centersColorsTextureSize[0], height: centersColorsTextureSize[1]},
            format: "rgba32uint",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
        });

        {
            let commandEncoder = engine_ctx.device.createCommandEncoder();
            commandEncoder.copyBufferToTexture({buffer: stageBuf_Covariances, bytesPerRow: covariancesTextureSize[0]*4}, {texture: this.covariancesTexture}, { width: covariancesTextureSize[0], height: covariancesTextureSize[1]});
            commandEncoder.copyBufferToTexture({buffer: stageBuf_CenterColors, bytesPerRow: centersColorsTextureSize[0]*16}, {texture: this.centersColorsTexture}, { width: centersColorsTextureSize[0], height: centersColorsTextureSize[1]});
            let cmdBuf = commandEncoder.finish();
            engine_ctx.queue.submit([cmdBuf]);
        }

        this.dConstant = engine_ctx.createBuffer0(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

        this.dIndices =  engine_ctx.createBuffer0(splatCount * 4, GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE);
        this.dIndices1 = engine_ctx.createBuffer0(splatCount * 4, GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE);        

        this.dDrawIndirect = engine_ctx.createBuffer0(16, GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST);

        {
            const uniform = new Uint32Array(4);
            uniform[0] = 6;
            uniform[1] = splatCount;
            uniform[2] = 0;
            uniform[3] = 0;
            engine_ctx.queue.writeBuffer(this.dDrawIndirect, 0, uniform.buffer, uniform.byteOffset, uniform.byteLength);
        }

        this.dConstantViewVec = engine_ctx.createBuffer0(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        this.dKey = [];
        
        let buf_size = splatCount;    
        {
            let buf = engine_ctx.createBuffer0(buf_size * 4, GPUBufferUsage.STORAGE);
            this.dKey.push(buf);
        }
        while(buf_size>1)
        {
            buf_size = Math.floor((buf_size + workgroup_size_2x - 1)/workgroup_size_2x);
            let buf = engine_ctx.createBuffer0(buf_size*4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
            this.dKey.push(buf);
        }

        this.dKeyMinMax = engine_ctx.createBuffer0(16, GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM);
        this.dConstantSort = engine_ctx.createBuffer0(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

        this.dBufScan1 =[];
        this.dBufScan2 =[];
        this.dBufScanSizes = [];
        buf_size = splatCount;
        while (buf_size>0)
        {
            let buf1 = engine_ctx.createBuffer0(buf_size * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
            let buf2 = engine_ctx.createBuffer0(buf_size * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
            this.dBufScan1.push(buf1);
            this.dBufScan2.push(buf2);
            this.dBufScanSizes.push(buf_size);
            buf_size = Math.floor((buf_size + workgroup_size_2x - 1)/workgroup_size_2x) - 1;
        }

        //////////////////////////////////////////////////////////

        if (!("render_splats" in  engine_ctx.cache.bindGroupLayouts))
        {
            engine_ctx.cache.bindGroupLayouts.render_splats = engine_ctx.device.createBindGroupLayout({
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.VERTEX,
                        texture:{
                            viewDimension: "2d",
                        }               
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.VERTEX,
                        texture:{
                            viewDimension: "2d",
                            sampleType: "uint"
                        }               
                    },
                    {
                        binding: 2,
                        visibility: GPUShaderStage.VERTEX,
                        buffer:{
                            type: "uniform"
                        }
                    },
                ]
            });
        }

        if (!("calc_key" in engine_ctx.cache.bindGroupLayouts))
        {
            engine_ctx.cache.bindGroupLayouts.calc_key = engine_ctx.device.createBindGroupLayout({
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer:{
                            type: "uniform"
                        }
                    
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        texture:{
                            viewDimension: "2d",
                            sampleType: "uint"
                        }         
                    },
                    {
                        binding: 2,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer:{
                            type: "storage"
                        }
                    }
                ]
            });
        }

        if (!("key_reduction" in engine_ctx.cache.bindGroupLayouts))
        {
            engine_ctx.cache.bindGroupLayouts.key_reduction = engine_ctx.device.createBindGroupLayout({
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer:{
                            type: "read-only-storage"
                        }
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer:{
                            type: "storage"
                        }
                    }
                ]
            });
        }


        if (!("reset_indices" in engine_ctx.cache.bindGroupLayouts))
        {
            engine_ctx.cache.bindGroupLayouts.reset_indices = engine_ctx.device.createBindGroupLayout({
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer:{
                            type: "storage"
                        }
                    }
                ]
            });
        }

        let layout_name = this.dBufScan1.length>1 ? "radixScan1B" : "radixScan1A";
        if (!(layout_name in engine_ctx.cache.bindGroupLayouts))
        {
            let layout_entries_radix_scan1 = [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "uniform"
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "uniform"
                    }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "read-only-storage"
                    }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "read-only-storage"
                    }
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "storage"
                    }
                },
                {
                    binding: 5,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "storage"
                    }
                },
            ];

            if (this.dBufScan1.length>1)
            {
                layout_entries_radix_scan1.push({
                    binding: 6,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "storage"
                    }
                });
    
                layout_entries_radix_scan1.push({
                    binding: 7,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "storage"
                    }
                });
            }
            engine_ctx.cache.bindGroupLayouts[layout_name] = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_radix_scan1 });
        }
        let bindGroupLayoutRadixScan1 = engine_ctx.cache.bindGroupLayouts[layout_name];


        layout_name = this.dBufScan1.length>1 ? "frustumScan1B" : "frustumScan1A";
        if (!(layout_name in engine_ctx.cache.bindGroupLayouts))
        {
            let layout_entries_frustum_scan1 = [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    texture:{
                        viewDimension: "2d",
                        sampleType: "uint"
                    }         
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "read-only-storage"
                    }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "storage"
                    }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "storage"
                    }
                },
            ];
            if (this.dBufScan1.length>1)
            {
                layout_entries_frustum_scan1.push({
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "storage"
                    }
                });
    
                layout_entries_frustum_scan1.push({
                    binding: 5,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "storage"
                    }
                });
            }
            engine_ctx.cache.bindGroupLayouts[layout_name] = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_frustum_scan1 });
        }
        let bindGroupLayoutFrustumScan1 = engine_ctx.cache.bindGroupLayouts[layout_name]; 

        if (!("radixScan2A" in engine_ctx.cache.bindGroupLayouts))
        {
            let layout_entries_radix_scan2 = [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "storage"
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "storage"
                    }
                },
            ];
    
            engine_ctx.cache.bindGroupLayouts.radixScan2A =engine_ctx.device.createBindGroupLayout({ entries: layout_entries_radix_scan2 });    
    
            layout_entries_radix_scan2.push({
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "storage"
                }
            });
        
            layout_entries_radix_scan2.push({
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "storage"
                }
            });
    
            engine_ctx.cache.bindGroupLayouts.radixScan2B = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_radix_scan2 });
        }

        if (!("radixScan3" in engine_ctx.cache.bindGroupLayouts))
        {
            engine_ctx.cache.bindGroupLayouts.radixScan3 = engine_ctx.device.createBindGroupLayout({ 
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer:{
                            type: "storage"
                        }
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer:{
                            type: "storage"
                        }
                    },
                    {
                        binding: 2,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer:{
                            type: "read-only-storage"
                        }
                    },
                    {
                        binding: 3,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer:{
                            type: "read-only-storage"
                        }
                    }
                ]                    
            });
        }

        if (!("radixScatter" in engine_ctx.cache.bindGroupLayouts))
        {
            engine_ctx.cache.bindGroupLayouts.radixScatter =  engine_ctx.device.createBindGroupLayout({ 
                entries: [  
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer:{
                            type: "read-only-storage"
                        }
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer:{
                            type: "read-only-storage"
                        }
                    },
                    {
                        binding: 2,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer:{
                            type: "read-only-storage"
                        }
                    },
                    {
                        binding: 3,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer:{
                            type: "storage"
                        }
                    }
                ]
            });
        }        

        //////////////////////////////////////////////////////////

        let covariancesView = this.covariancesTexture.createView();
        let centersColorsView = this.centersColorsTexture.createView();

        let group_entries_render_splats = [
            {
                binding: 0,
                resource: covariancesView
            },
            {
                binding: 1,
                resource: centersColorsView
            },
            {
                binding: 2,
                resource:{
                    buffer: this.dConstant            
                }
            }

        ];
        this.bind_group_render_splats = engine_ctx.device.createBindGroup({ layout: engine_ctx.cache.bindGroupLayouts.render_splats, entries: group_entries_render_splats});

        let group_entries_calc_key = [
            {
                binding: 0,
                resource:{
                    buffer: this.dConstantViewVec            
                }
            },
            {
                binding: 1,
                resource: centersColorsView
            },
            {
                binding: 2,
                resource:{
                    buffer: this.dKey[0]
                }
            },
        ];

        this.bind_group_calc_key = engine_ctx.device.createBindGroup({ layout: engine_ctx.cache.bindGroupLayouts.calc_key, entries: group_entries_calc_key});

        this.bind_group_key_reduction = [];
        for (let i=0; i<this.dKey.length-1; i++)
        {
            let group_entries = [
                {
                    binding: 0,
                    resource:{
                        buffer: this.dKey[i]            
                    }
                },
                {
                    binding: 1,
                    resource:{
                        buffer: this.dKey[i+1]
                    }
                }
            ];
            this.bind_group_key_reduction.push(engine_ctx.device.createBindGroup({ layout: engine_ctx.cache.bindGroupLayouts.key_reduction, entries: group_entries}));
        }

        let group_entries_reset_indices = [
            {
                binding: 0,
                resource:{
                    buffer: this.dIndices            
                }
            },           
        ];

        this.bind_group_reset_indices = engine_ctx.device.createBindGroup({ layout: engine_ctx.cache.bindGroupLayouts.reset_indices, entries: group_entries_reset_indices});

        {
            let commandEncoder = engine_ctx.device.createCommandEncoder();  
            ResetIndices(commandEncoder, this);
            let cmdBuf = commandEncoder.finish();
            engine_ctx.queue.submit([cmdBuf]);
        }
      

        let group_entries_frustum_scan1 = [
            {
                binding: 0,
                resource: centersColorsView
            },
            {
                binding: 1,
                resource:{
                    buffer: this.dIndices
                }
            },
            {
                binding: 2,
                resource:{
                    buffer: this.dBufScan1[0]
                }
            },
            {
                binding: 3,
                resource:{
                    buffer: this.dBufScan2[0]
                }
            },
        ];

        if (this.dBufScan1.length>1)
        {
            group_entries_frustum_scan1.push({
                binding: 4,
                resource:{
                    buffer: this.dBufScan1[1]
                }
            });

            group_entries_frustum_scan1.push({
                binding: 5,
                resource:{
                    buffer: this.dBufScan2[1]
                }
            });
        }
        this.bind_group_frustum_scan1 = engine_ctx.device.createBindGroup({ layout: bindGroupLayoutFrustumScan1, entries: group_entries_frustum_scan1}); 

        this.bind_group_radix_scan1 = new Array(2);
        this.bind_group_radix_scatter = new Array(2);

        let group_entries_radix_scan10 = [
            {
                binding: 0,
                resource:{
                    buffer: this.dConstantSort
                }
            },
            {
                binding: 1,
                resource:{
                    buffer: this.dKeyMinMax
                }
            },
            {
                binding: 2,
                resource:{
                    buffer: this.dKey[0]
                }
            },
            {
                binding: 3,
                resource:{
                    buffer: this.dIndices
                }
            },
            {
                binding: 4,
                resource:{
                    buffer: this.dBufScan1[0]
                }
            },
            {
                binding: 5,
                resource:{
                    buffer: this.dBufScan2[0]
                }
            },
        ];

        if (this.dBufScan1.length>1)
        {
            group_entries_radix_scan10.push({
                binding: 6,
                resource:{
                    buffer: this.dBufScan1[1]
                }
            });

            group_entries_radix_scan10.push({
                binding: 7,
                resource:{
                    buffer: this.dBufScan2[1]
                }
            });
        }

        this.bind_group_radix_scan1[0] = engine_ctx.device.createBindGroup({ layout: bindGroupLayoutRadixScan1, entries: group_entries_radix_scan10}); 

        let group_entries_radix_scatter0 = [
            {
                binding: 0,
                resource:{
                    buffer: this.dIndices    
                }
            },
            {
                binding: 1,
                resource:{
                    buffer: this.dBufScan1[0]
                }
            },
            {
                binding: 2,
                resource:{
                    buffer: this.dBufScan2[0]
                }
            },
            {
                binding: 3,
                resource:{
                    buffer: this.dIndices1
                }
            },
        ];

        this.bind_group_radix_scatter[0] = engine_ctx.device.createBindGroup({ layout: engine_ctx.cache.bindGroupLayouts.radixScatter, entries: group_entries_radix_scatter0});

        let group_entries_radix_scan11 = [
            {
                binding: 0,
                resource:{
                    buffer: this.dConstantSort
                }
            },
            {
                binding: 1,
                resource:{
                    buffer: this.dKeyMinMax
                }
            },
            {
                binding: 2,
                resource:{
                    buffer: this.dKey[0]
                }
            },
            {
                binding: 3,
                resource:{
                    buffer:  this.dIndices1
                }
            },
            {
                binding: 4,
                resource:{
                    buffer: this.dBufScan1[0]
                }
            },
            {
                binding: 5,
                resource:{
                    buffer: this.dBufScan2[0]
                }
            },
        ];

        if (this.dBufScan1.length>1)
        {
            group_entries_radix_scan11.push({
                binding: 6,
                resource:{
                    buffer: this.dBufScan1[1]
                }
            });

            group_entries_radix_scan11.push({
                binding: 7,
                resource:{
                    buffer: this.dBufScan2[1]
                }
            });
        }

        this.bind_group_radix_scan1[1] = engine_ctx.device.createBindGroup({ layout: bindGroupLayoutRadixScan1, entries: group_entries_radix_scan11}); 

        let group_entries_radix_scatter1 = [
            {
                binding: 0,
                resource:{
                    buffer: this.dIndices1    
                }
            },
            {
                binding: 1,
                resource:{
                    buffer: this.dBufScan1[0]
                }
            },
            {
                binding: 2,
                resource:{
                    buffer: this.dBufScan2[0]
                }
            },
            {
                binding: 3,
                resource:{
                    buffer: this.dIndices
                }
            },
        ];

        this.bind_group_radix_scatter[1] = engine_ctx.device.createBindGroup({ layout: engine_ctx.cache.bindGroupLayouts.radixScatter, entries: group_entries_radix_scatter1});

        this.bind_group_radix_scan2 = [];
        for (let i=1; i<this.dBufScan1.length; i++)
        {
            let group_entries_radix_scan = [            
                {
                    binding: 0,
                    resource:{
                        buffer: this.dBufScan1[i]
                    }
                },
                {
                    binding: 1,
                    resource:{
                        buffer: this.dBufScan2[i]
                    }
                },
            ];

            if (i<this.dBufScan1.length-1)
            {
                group_entries_radix_scan.push({
                    binding: 2,
                    resource:{
                        buffer: this.dBufScan1[i+1]
                    }
                });
    
                group_entries_radix_scan.push({
                    binding: 3,
                    resource:{
                        buffer: this.dBufScan2[i+1]
                    }
                });

                this.bind_group_radix_scan2.push(engine_ctx.device.createBindGroup({ layout:  engine_ctx.cache.bindGroupLayouts.radixScan2B, entries: group_entries_radix_scan}));
            }
            else
            {
                this.bind_group_radix_scan2.push(engine_ctx.device.createBindGroup({ layout:  engine_ctx.cache.bindGroupLayouts.radixScan2A, entries: group_entries_radix_scan}));
            }
        }

        this.bind_group_radix_scan3 = [];
        for (let i=0; i < this.dBufScan1.length - 1; i++)
        {
            let group_entries_radix_scan = [            
                {
                    binding: 0,
                    resource:{
                        buffer: this.dBufScan1[i]
                    }
                },
                {
                    binding: 1,
                    resource:{
                        buffer: this.dBufScan2[i]
                    }
                },
                {
                    binding: 2,
                    resource:{
                        buffer: this.dBufScan1[i + 1]
                    }
                },
                {
                    binding: 3,
                    resource:{
                        buffer: this.dBufScan2[i + 1]
                    }
                }
            ];
            this.bind_group_radix_scan3.push(engine_ctx.device.createBindGroup({ layout:  engine_ctx.cache.bindGroupLayouts.radixScan3, entries: group_entries_radix_scan}));
        }

    }

    updateConstant(camera, devicePixelRatio, renderDimensions)
    {
        const uniform = new Float32Array(8);
        uniform[0] = camera.projectionMatrix.elements[0] * devicePixelRatio * renderDimensions.x * 0.45;
        uniform[1] = camera.projectionMatrix.elements[5] * devicePixelRatio * renderDimensions.y * 0.45;
        uniform[2] = 2.0 / (renderDimensions.x * devicePixelRatio);
        uniform[3] = 2.0 / (renderDimensions.y * devicePixelRatio);   
        uniform[4] = this.r_aperture;
        uniform[5] = this.focal_z;     
        engine_ctx.queue.writeBuffer(this.dConstant, 0, uniform.buffer, uniform.byteOffset, uniform.byteLength);
    }

    frustumCull(camera)
    {
        FrustumCull(camera, this);
    }

    sort(viewVec)
    {
        {
            const uniform = new Float32Array(4);
            uniform[0] = viewVec.x;
            uniform[1] = viewVec.y;
            uniform[2] = viewVec.z;
            engine_ctx.queue.writeBuffer(this.dConstantViewVec, 0, uniform.buffer, uniform.byteOffset, uniform.byteLength);
        }

        let commandEncoder = engine_ctx.device.createCommandEncoder();    
        CalcKey(commandEncoder, this);
        KeyReduction(commandEncoder, this);
        let cmdBuf = commandEncoder.finish();
        engine_ctx.queue.submit([cmdBuf]);

        RadixSort(this);
    }

}

