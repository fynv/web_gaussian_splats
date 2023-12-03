const workgroup_size = 64;
const workgroup_size_2x = workgroup_size * 2;

function condition(cond, a, b="")
{
    return cond? a: b;
}

function get_shader_frustum_scan1(has_group_buf)
{
    return  `
struct Camera
{
    projMat: mat4x4f, 
    viewMat: mat4x4f,
    invProjMat: mat4x4f,
    invViewMat: mat4x4f,
    eyePos: vec4f,
    scissor: vec4f
};

@group(0) @binding(0)
var<uniform> uCamera: Camera;

@group(1) @binding(0)
var uCentersColorsTexture: texture_2d<u32>;

@group(1) @binding(1)
var<storage, read> bIndices : array<u32>;

@group(1) @binding(2)
var<storage, read_write> bData1 : array<i32>;    

@group(1) @binding(3)
var<storage, read_write> bData2 : array<i32>;    

${condition(has_group_buf,`
@group(1) @binding(4)
var<storage, read_write> bGroup1 : array<i32>;

@group(1) @binding(5)
var<storage, read_write> bGroup2 : array<i32>;
`)}


const encodeNorm4 = vec4(1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0); 
const mask4 = vec4(u32(0x000000FF), u32(0x0000FF00), u32(0x00FF0000), u32(0xFF000000));
const shift4 = vec4u(0, 8, 16, 24);

fn uintToRGBAVec (u: u32) ->vec4f
{
    let urgba = (mask4 & vec4(u)) >> vec4(shift4);
    return vec4f(urgba) * encodeNorm4;
}

fn getDataIdx(idx: u32, stride: i32, offset: i32, width: u32) -> vec2u
{    
    let d = idx * u32(stride) + u32(offset);
    return vec2(d%width, d/width);
}

var<workgroup> s_buf1 : array<i32, ${workgroup_size_2x}>;
var<workgroup> s_buf2 : array<i32, ${workgroup_size_2x}>;

fn inFrustum(idx : u32, width: u32) -> bool
{
    let sampledCenterColor = textureLoad(uCentersColorsTexture, getDataIdx(idx, 1, 0, width), 0);
    let splatCenter = bitcast<vec3f>(sampledCenterColor.yzw);
    let color = uintToRGBAVec(sampledCenterColor.x);
    if (color.w == 0.0)
    {
        return false;
    }

    let viewCenter = uCamera.viewMat * vec4(splatCenter, 1.0);
    if (length(viewCenter)>125.0)
    {
        return false;
    }
    let clipCenter = uCamera.projMat * viewCenter;
    let ndcCenter = clipCenter.xyz / clipCenter.w;                
    return ndcCenter.x>-1.0 && ndcCenter.x<1.0 && ndcCenter.y>-1.0 && ndcCenter.y<1.0 && ndcCenter.z>-1.0 && ndcCenter.z<1.0;
}

@compute @workgroup_size(${workgroup_size},1,1)
fn main(
    @builtin(local_invocation_id) LocalInvocationID : vec3<u32>,
    @builtin(workgroup_id) WorkgroupID : vec3<u32>)
{
    let threadIdx = LocalInvocationID.x;
    let blockIdx = WorkgroupID.x;    
    let count = arrayLength(&bData1);

    let width_CC = textureDimensions(uCentersColorsTexture).x;

    var i = threadIdx + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        let j = bIndices[i];                      
        let pred = !inFrustum(j, width_CC);
        s_buf1[threadIdx] = select(1,0,pred);
        s_buf2[threadIdx] = select(0,1,pred);
    }

    i = threadIdx + ${workgroup_size} + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        let j = bIndices[i];
        let pred = !inFrustum(j, width_CC);
        s_buf1[threadIdx + ${workgroup_size}] = select(1,0,pred);
        s_buf2[threadIdx + ${workgroup_size}] = select(0,1,pred);
    }

    workgroupBarrier();

    var half_size_group = 1u;
    var size_group = 2u;

    while(half_size_group <= ${workgroup_size})
    {
        let gid = threadIdx/half_size_group;
        let tid = gid*size_group + half_size_group + threadIdx % half_size_group;
        i = tid + blockIdx*${workgroup_size_2x};
        if (i<count)
        {
            s_buf1[tid] = s_buf1[gid*size_group + half_size_group -1] + s_buf1[tid];
            s_buf2[tid] = s_buf2[gid*size_group + half_size_group -1] + s_buf2[tid];
        }
        half_size_group = half_size_group << 1;
        size_group = size_group << 1;
        workgroupBarrier();
    }

    i = threadIdx + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        bData1[i] = s_buf1[threadIdx];
        bData2[i] = s_buf2[threadIdx];
    }
    
    i = threadIdx + ${workgroup_size} + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        bData1[i] = s_buf1[threadIdx + ${workgroup_size}];
        bData2[i] = s_buf2[threadIdx + ${workgroup_size}];
    }

${condition(has_group_buf,`        
    let count_group = arrayLength(&bGroup1);
    if (threadIdx == 0 && blockIdx<count_group)
    {        
        bGroup1[blockIdx] = s_buf1[${workgroup_size_2x} - 1];
        bGroup2[blockIdx] = s_buf2[${workgroup_size_2x} - 1];
    }
`)}
}
`;
}

function GetPipelineFrustumScan1(has_group_buf)
{
    let name = has_group_buf? "frustumScan1B" : "frustumScan1A";
    if (!(name in engine_ctx.cache.pipelines))
    {
        let camera_options = { has_reflector: false };
        let camera_signature =  JSON.stringify(camera_options);
        let camera_layout = engine_ctx.cache.bindGroupLayouts.perspective_camera[camera_signature];

        let shaderModule = engine_ctx.device.createShaderModule({ code: get_shader_frustum_scan1(has_group_buf) });
        let bindGroupLayouts = [camera_layout, engine_ctx.cache.bindGroupLayouts[name]];
        const pipelineLayoutDesc = { bindGroupLayouts };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);

        engine_ctx.cache.pipelines[name] = engine_ctx.device.createComputePipeline({
            layout,
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
    }
    return engine_ctx.cache.pipelines[name];
}

function get_shader_radix_scan1(has_group_buf)
{
    return  `
@group(0) @binding(0)
var<uniform> uBit: u32;

@group(0) @binding(1)
var<uniform> uMinMax: vec2f;

@group(0) @binding(2)
var<storage, read> bKey : array<f32>;

@group(0) @binding(3)
var<storage, read> bIndices : array<u32>;

@group(0) @binding(4)
var<storage, read_write> bData1 : array<i32>;    

@group(0) @binding(5)
var<storage, read_write> bData2 : array<i32>;    

${condition(has_group_buf,`
@group(0) @binding(6)
var<storage, read_write> bGroup1 : array<i32>;

@group(0) @binding(7)
var<storage, read_write> bGroup2 : array<i32>;
`)}

var<workgroup> s_buf1 : array<i32, ${workgroup_size_2x}>;
var<workgroup> s_buf2 : array<i32, ${workgroup_size_2x}>;

@compute @workgroup_size(${workgroup_size},1,1)
fn main(
    @builtin(local_invocation_id) LocalInvocationID : vec3<u32>,
    @builtin(workgroup_id) WorkgroupID : vec3<u32>)
{
    let threadIdx = LocalInvocationID.x;
    let blockIdx = WorkgroupID.x;    
    let count = arrayLength(&bData1);

    var i = threadIdx + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        let j = bIndices[i];
        let key = bKey[j];
        let q = u32((key - uMinMax.x)/(uMinMax.y - uMinMax.x) * 65535.0);        
        let pred = (q & (1u << uBit)) != 0;
        s_buf1[threadIdx] = select(1,0,pred);
        s_buf2[threadIdx] = select(0,1,pred);
    }

    i = threadIdx + ${workgroup_size} + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        let j = bIndices[i];
        let key = bKey[j];
        let q = u32((key - uMinMax.x)/(uMinMax.y - uMinMax.x) * 65535.0);
        let pred = (q & (1u << uBit)) != 0;
        s_buf1[threadIdx + ${workgroup_size}] = select(1,0,pred);
        s_buf2[threadIdx + ${workgroup_size}] = select(0,1,pred);
    }

    workgroupBarrier();

    var half_size_group = 1u;
    var size_group = 2u;

    while(half_size_group <= ${workgroup_size})
    {
        let gid = threadIdx/half_size_group;
        let tid = gid*size_group + half_size_group + threadIdx % half_size_group;
        i = tid + blockIdx*${workgroup_size_2x};
        if (i<count)
        {
            s_buf1[tid] = s_buf1[gid*size_group + half_size_group -1] + s_buf1[tid];
            s_buf2[tid] = s_buf2[gid*size_group + half_size_group -1] + s_buf2[tid];
        }
        half_size_group = half_size_group << 1;
        size_group = size_group << 1;
        workgroupBarrier();
    }

    i = threadIdx + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        bData1[i] = s_buf1[threadIdx];
        bData2[i] = s_buf2[threadIdx];
    }
    
    i = threadIdx + ${workgroup_size} + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        bData1[i] = s_buf1[threadIdx + ${workgroup_size}];
        bData2[i] = s_buf2[threadIdx + ${workgroup_size}];
    }

${condition(has_group_buf,`        
    let count_group = arrayLength(&bGroup1);
    if (threadIdx == 0 && blockIdx<count_group)
    {        
        bGroup1[blockIdx] = s_buf1[${workgroup_size_2x} - 1];
        bGroup2[blockIdx] = s_buf2[${workgroup_size_2x} - 1];
    }
`)}
}
`;
}

function GetPipelineRadixScan1(has_group_buf)
{
    let name = has_group_buf? "radixScan1B" : "radixScan1A";
    if (!(name in engine_ctx.cache.pipelines))
    {
        let shaderModule = engine_ctx.device.createShaderModule({ code: get_shader_radix_scan1(has_group_buf) });
        let bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts[name]];
        const pipelineLayoutDesc = { bindGroupLayouts };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);

        engine_ctx.cache.pipelines[name] = engine_ctx.device.createComputePipeline({
            layout,
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
    }
    return engine_ctx.cache.pipelines[name];
}

function get_shader_radix_scan2(has_group_buf)
{
    return  `    
@group(0) @binding(0)
var<storage, read_write> bData1 : array<i32>;    

@group(0) @binding(1)
var<storage, read_write> bData2 : array<i32>;    

${condition(has_group_buf,`
@group(0) @binding(2)
var<storage, read_write> bGroup1 : array<i32>;

@group(0) @binding(3)
var<storage, read_write> bGroup2 : array<i32>;
`)}

var<workgroup> s_buf1 : array<i32, ${workgroup_size_2x}>;
var<workgroup> s_buf2 : array<i32, ${workgroup_size_2x}>;

@compute @workgroup_size(${workgroup_size},1,1)
fn main(
    @builtin(local_invocation_id) LocalInvocationID : vec3<u32>,
    @builtin(workgroup_id) WorkgroupID : vec3<u32>)
{
    let threadIdx = LocalInvocationID.x;
    let blockIdx = WorkgroupID.x;    
    let count = arrayLength(&bData1);

    var i = threadIdx + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        s_buf1[threadIdx] = bData1[i];
        s_buf2[threadIdx] = bData2[i];
    }

    i = threadIdx + ${workgroup_size} + blockIdx*${workgroup_size_2x};
    if (i<count)
    {        
        s_buf1[threadIdx + ${workgroup_size}] = bData1[i];
        s_buf2[threadIdx + ${workgroup_size}] = bData2[i];
    }

    workgroupBarrier();

    var half_size_group = 1u;
    var size_group = 2u;

    while(half_size_group <= ${workgroup_size})
    {
        let gid = threadIdx/half_size_group;
        let tid = gid*size_group + half_size_group + threadIdx % half_size_group;
        i = tid + blockIdx*${workgroup_size_2x};
        if (i<count)
        {
            s_buf1[tid] = s_buf1[gid*size_group + half_size_group -1] + s_buf1[tid];
            s_buf2[tid] = s_buf2[gid*size_group + half_size_group -1] + s_buf2[tid];
        }
        half_size_group = half_size_group << 1;
        size_group = size_group << 1;
        workgroupBarrier();
    }

    i = threadIdx + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        bData1[i] = s_buf1[threadIdx];
        bData2[i] = s_buf2[threadIdx];
    }
    
    i = threadIdx + ${workgroup_size} + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        bData1[i] = s_buf1[threadIdx + ${workgroup_size}];
        bData2[i] = s_buf2[threadIdx + ${workgroup_size}];
    }

${condition(has_group_buf,`        
    let count_group = arrayLength(&bGroup1);
    if (threadIdx == 0 && blockIdx<count_group)
    {        
        bGroup1[blockIdx] = s_buf1[${workgroup_size_2x} - 1];
        bGroup2[blockIdx] = s_buf2[${workgroup_size_2x} - 1];
    }
`)}    

}
`;
}

function GetPipelineRadixScan2(has_group_buf)
{
    let name = has_group_buf? "radixScan2B" : "radixScan2A";
    if (!(name in engine_ctx.cache.pipelines))
    {
        let shaderModule = engine_ctx.device.createShaderModule({ code: get_shader_radix_scan2(has_group_buf) });
        let bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts[name]];
        const pipelineLayoutDesc = { bindGroupLayouts };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);

        engine_ctx.cache.pipelines[name] = engine_ctx.device.createComputePipeline({
            layout,
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
    }
    return engine_ctx.cache.pipelines[name];
}

function get_shader_radix_scan3()
{
    return  ` 
@group(0) @binding(0)
var<storage, read_write> bData1 : array<i32>;    

@group(0) @binding(1)
var<storage, read_write> bData2 : array<i32>;    

@group(0) @binding(2)
var<storage, read> bGroup1 : array<i32>;

@group(0) @binding(3)
var<storage, read> bGroup2 : array<i32>;

@compute @workgroup_size(${workgroup_size},1,1)
fn main(
    @builtin(local_invocation_id) LocalInvocationID : vec3<u32>,
    @builtin(workgroup_id) WorkgroupID : vec3<u32>)
{
    let threadIdx = LocalInvocationID.x;
    let blockIdx = WorkgroupID.x + 2;        

    let add_idx = WorkgroupID.x / 2;
    let i = threadIdx + blockIdx*${workgroup_size};

    {
        let value = bData1[i];
        bData1[i] = value + bGroup1[add_idx];
    }

    {
        let value = bData2[i];
        bData2[i] = value + bGroup2[add_idx];
    }
}
`;
}

function GetPipelineRadixScan3()
{
    if (!("radixScan3" in engine_ctx.cache.pipelines))
    {
        let shaderModule = engine_ctx.device.createShaderModule({ code: get_shader_radix_scan3() });
        let bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.radixScan3];
        const pipelineLayoutDesc = { bindGroupLayouts };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);

        engine_ctx.cache.pipelines.radixScan3 = engine_ctx.device.createComputePipeline({
            layout,
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
    }
    return engine_ctx.cache.pipelines.radixScan3;
}

function get_shader_radix_scatter()
{
    return  `
@group(0) @binding(0)
var<storage, read> bInput : array<u32>;

@group(0) @binding(1)
var<storage, read> bIndices1 : array<i32>;

@group(0) @binding(2)
var<storage, read> bIndices2 : array<i32>;

@group(0) @binding(3)
var<storage, read_write> bOutput : array<u32>;

@compute @workgroup_size(${workgroup_size},1,1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>)
{
    let count = arrayLength(&bInput);
    let idx = GlobalInvocationID.x;
    if (idx>=count)
    {
        return;
    }    
    let value = bInput[idx];
    if ((idx == 0 && bIndices1[idx]>0) || (idx > 0 && bIndices1[idx]>bIndices1[idx-1]))
    {
        bOutput[bIndices1[idx] - 1] = value;
    }
    else
    {
        let count0 = bIndices1[count -1];
        bOutput[count0 + bIndices2[idx] - 1] = value;
    }
}
`
}

function GetPipelineRadixScatter()
{
    if (!("radixScatter" in engine_ctx.cache.pipelines))
    {
        let shaderModule = engine_ctx.device.createShaderModule({ code: get_shader_radix_scatter() });
        let bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.radixScatter];
        const pipelineLayoutDesc = { bindGroupLayouts };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);

        engine_ctx.cache.pipelines.radixScatter = engine_ctx.device.createComputePipeline({
            layout,
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
    }
    return engine_ctx.cache.pipelines.radixScatter;
}

export function FrustumCull(camera, splats)
{
    const splatCount = splats.splatBuffer.getSplatCount();

    let commandEncoder = engine_ctx.device.createCommandEncoder();    

    {            
        const passEncoder = commandEncoder.beginComputePass();
        {
            let num_groups = Math.floor((splatCount + workgroup_size_2x - 1)/workgroup_size_2x); 
            let pipeline =  GetPipelineFrustumScan1(splats.dBufScan1.length>1);
            passEncoder.setPipeline(pipeline);
            passEncoder.setBindGroup(0, camera.bind_group);
            passEncoder.setBindGroup(1, splats.bind_group_frustum_scan1);
            passEncoder.dispatchWorkgroups(num_groups, 1,1); 
        }            
        
        for (let k = 0; k<splats.bind_group_radix_scan2.length; k++)
        {
            let num_groups = Math.floor((splats.dBufScanSizes[k+1] + workgroup_size_2x - 1)/workgroup_size_2x); 
            let pipeline =  GetPipelineRadixScan2(k<splats.bind_group_radix_scan2.length - 1);
            passEncoder.setPipeline(pipeline);                
            passEncoder.setBindGroup(0, splats.bind_group_radix_scan2[k]);
            passEncoder.dispatchWorkgroups(num_groups, 1,1); 
        }

        for (let k = splats.bind_group_radix_scan3.length - 1; k>=0; k--)
        {
            let num_groups = Math.floor((splats.dBufScanSizes[k] + workgroup_size - 1)/workgroup_size) - 2; 
            let pipeline =  GetPipelineRadixScan3();
            passEncoder.setPipeline(pipeline);
            passEncoder.setBindGroup(0, splats.bind_group_radix_scan3[k]);
            passEncoder.dispatchWorkgroups(num_groups, 1,1);
        }

        passEncoder.end();
    }

    {
        let num_groups = Math.floor((splatCount + workgroup_size -1)/workgroup_size);
        let pipeline =  GetPipelineRadixScatter();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, splats.bind_group_radix_scatter[0]);
        passEncoder.dispatchWorkgroups(num_groups, 1,1); 
        passEncoder.end();
    }

    commandEncoder.copyBufferToBuffer(splats.dBufScan1[0], (splatCount-1)*4, splats.dDrawIndirect, 4, 4);

    let cmdBuf = commandEncoder.finish();
    engine_ctx.queue.submit([cmdBuf]);

}


export function RadixSort(splats)
{
    const splatCount = splats.splatBuffer.getSplatCount();
    let bits = 16;

    for (let i=0; i<bits; i++)
    {
        {
            const uniform = new Int32Array(4);
            uniform[0] = i;
            engine_ctx.queue.writeBuffer(splats.dConstantSort, 0, uniform.buffer, uniform.byteOffset, uniform.byteLength);
        }

        let commandEncoder = engine_ctx.device.createCommandEncoder();    

        let j = i % 2;   
      

        {            
            const passEncoder = commandEncoder.beginComputePass();
            {
                let num_groups = Math.floor((splatCount + workgroup_size_2x - 1)/workgroup_size_2x); 
                let pipeline =  GetPipelineRadixScan1(splats.dBufScan1.length>1);
                passEncoder.setPipeline(pipeline);
                passEncoder.setBindGroup(0, splats.bind_group_radix_scan1[j]);
                passEncoder.dispatchWorkgroups(num_groups, 1,1); 
            }            
            
            for (let k = 0; k<splats.bind_group_radix_scan2.length; k++)
            {
                let num_groups = Math.floor((splats.dBufScanSizes[k+1] + workgroup_size_2x - 1)/workgroup_size_2x); 
                let pipeline =  GetPipelineRadixScan2(k<splats.bind_group_radix_scan2.length - 1);
                passEncoder.setPipeline(pipeline);                
                passEncoder.setBindGroup(0, splats.bind_group_radix_scan2[k]);
                passEncoder.dispatchWorkgroups(num_groups, 1,1); 
            }

            for (let k = splats.bind_group_radix_scan3.length - 1; k>=0; k--)
            {
                let num_groups = Math.floor((splats.dBufScanSizes[k] + workgroup_size - 1)/workgroup_size) - 2; 
                let pipeline =  GetPipelineRadixScan3();
                passEncoder.setPipeline(pipeline);
                passEncoder.setBindGroup(0, splats.bind_group_radix_scan3[k]);
                passEncoder.dispatchWorkgroups(num_groups, 1,1);
            }

            passEncoder.end();
        }


        {
            let num_groups = Math.floor((splatCount + workgroup_size -1)/workgroup_size);
            let pipeline =  GetPipelineRadixScatter();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(pipeline);
            passEncoder.setBindGroup(0, splats.bind_group_radix_scatter[j]);
            passEncoder.dispatchWorkgroups(num_groups, 1,1); 
            passEncoder.end();
        }

        let cmdBuf = commandEncoder.finish();
        engine_ctx.queue.submit([cmdBuf]);

    }

}
