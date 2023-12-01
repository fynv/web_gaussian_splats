const shader_code =`
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
var uCovariancesTexture: texture_2d<f32>;

@group(1) @binding(1)
var uCentersColorsTexture: texture_2d<u32>;

struct Constant
{
    focal: vec2f,
    basisViewport: vec2f
};

@group(1) @binding(2)
var<uniform> uRender: Constant;

struct VSIn 
{
    @builtin(vertex_index) vertId: u32,    
    @location(0) index : u32,    
};

struct VSOut 
{
    @builtin(position) Position: vec4f,
    @location(0) color: vec4f,
    @location(1) normPos: vec2f,    
};

const c_norm_pos = array(
    vec2(-1.0, -1.0),
    vec2(1.0, -1.0),
    vec2(-1.0, 1.0),
    vec2(1.0, 1.0),
    vec2(-1.0, 1.0),
    vec2(1.0, -1.0)
);

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

@vertex
fn vs_main(input: VSIn) -> VSOut
{
    var out: VSOut;
    out.normPos = c_norm_pos[input.vertId] * 2.0;

    let idx = input.index;

    let width_Cov = textureDimensions(uCovariancesTexture).x;
    let width_CC = textureDimensions(uCentersColorsTexture).x;
    
    let sampledCenterColor = textureLoad(uCentersColorsTexture, getDataIdx(idx, 1, 0, width_CC), 0);
    let splatCenter = bitcast<vec3f>(sampledCenterColor.yzw);
    out.color = uintToRGBAVec(sampledCenterColor.x);

    let viewCenter = uCamera.viewMat * vec4(splatCenter, 1.0);
    let clipCenter = uCamera.projMat * viewCenter;

    let sampledCovarianceA = textureLoad(uCovariancesTexture, getDataIdx(idx, 3, 0, width_Cov), 0).xy;
    let sampledCovarianceB = textureLoad(uCovariancesTexture, getDataIdx(idx, 3, 1, width_Cov), 0).xy;
    let sampledCovarianceC = textureLoad(uCovariancesTexture, getDataIdx(idx, 3, 2, width_Cov), 0).xy;

    let cov3D_M11_M12_M13 = vec3(sampledCovarianceA.rg, sampledCovarianceB.r);
    let cov3D_M22_M23_M33 = vec3(sampledCovarianceB.g, sampledCovarianceC.rg);

    let Vrk = mat3x3(
        cov3D_M11_M12_M13.x, cov3D_M11_M12_M13.y, cov3D_M11_M12_M13.z,
        cov3D_M11_M12_M13.y, cov3D_M22_M23_M33.x, cov3D_M22_M23_M33.y,
        cov3D_M11_M12_M13.z, cov3D_M22_M23_M33.y, cov3D_M22_M23_M33.z
    );

    let s = 1.0 / (viewCenter.z * viewCenter.z);   
    let focal = uRender.focal;
    let J = mat3x3(
        focal.x / viewCenter.z, 0.0, -(focal.x * viewCenter.x) * s,
        0.0, focal.y / viewCenter.z, -(focal.y * viewCenter.y) * s,
        0.0, 0.0, 0.0
    );

    let W = transpose(mat3x3(uCamera.viewMat[0].xyz, uCamera.viewMat[1].xyz, uCamera.viewMat[2].xyz));
    let T = W * J;
    var cov2Dm = transpose(T) * Vrk * T;
    cov2Dm[0][0] += 0.3;
    cov2Dm[1][1] += 0.3;

    let cov2Dv = vec3(cov2Dm[0][0], cov2Dm[0][1], cov2Dm[1][1]);
    let ndcCenter = clipCenter.xyz / clipCenter.w;

    let a = cov2Dv.x;
    let d = cov2Dv.z;
    let b = cov2Dv.y;
    let D = a * d - b * b;
    let trace = a + d;
    let traceOver2 = 0.5 * trace;
    let term2 = sqrt(trace * trace / 4.0 - D);
    let eigenValue1 = traceOver2 + term2;
    let eigenValue2 = max(traceOver2 - term2, 0.0); 

    const maxSplatSize = 1024.0;
    let eigenVector1 = normalize(vec2(b, eigenValue1 - a));
    
    let eigenVector2 = vec2(eigenVector1.y, -eigenVector1.x);
    let basisVector1 = eigenVector1 * min(sqrt(2.0 * eigenValue1), maxSplatSize);
    let basisVector2 = eigenVector2 * min(sqrt(2.0 * eigenValue2), maxSplatSize);

    let ndcOffset = vec2(out.normPos.x * basisVector1 + out.normPos.y * basisVector2) * uRender.basisViewport;

    out.Position = vec4(ndcCenter.xy + ndcOffset, (ndcCenter.z + 1.0)*0.5, 1.0);

    return out;
}

@fragment
fn fs_main(@location(0) color: vec4f, @location(1) normPos: vec2f) -> @location(0) vec4f
{ 
    var A = -dot(normPos, normPos);
    if (A < -4.0) 
    {
        discard;
    }
    A = exp(A) * color.w;
    return vec4(color.xyz * A, A);
}
`;


function GetPipeline(view_format, msaa)
{
    if (!("render_splats" in engine_ctx.cache.pipelines))
    {
        let camera_options = { has_reflector: false };
        let camera_signature =  JSON.stringify(camera_options);
        let camera_layout = engine_ctx.cache.bindGroupLayouts.perspective_camera[camera_signature];

        const pipelineLayoutDesc = { bindGroupLayouts: [camera_layout, engine_ctx.cache.bindGroupLayouts.render_splats] };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);
        let shaderModule = engine_ctx.device.createShaderModule({ code: shader_code });

        let vertex_bufs = [
            {            
                arrayStride: 4,
                stepMode: 'instance',
                attributes: [
                  {                
                    shaderLocation: 0,
                    offset: 0,
                    format: 'uint32',
                  },             
                ],
            }
        ];

        const vertex = {
            module: shaderModule,
            entryPoint: 'vs_main',
            buffers: vertex_bufs
        };

        const colorState = {
            format: view_format,                    
            writeMask: GPUColorWrite.ALL,
            blend: {
                color: {
                    srcFactor: "one",
                    dstFactor: "one-minus-src-alpha"
                },
                alpha: {
                    srcFactor: "one",
                    dstFactor: "one-minus-src-alpha"
                }
            }
        };

        const fragment = {
            module: shaderModule,
            entryPoint: 'fs_main',
            targets: [colorState]
        };

        const primitive = {
            frontFace: 'ccw',
            cullMode:  "none",
            topology: 'triangle-list'
        };

        const pipelineDesc = {
            layout,
    
            vertex,
            fragment,
    
            primitive
        };
        
        if (msaa)
        {
            pipelineDesc.multisample ={
                count: 4,
            };
        }

        engine_ctx.cache.pipelines.render_splats = engine_ctx.device.createRenderPipeline(pipelineDesc); 
        
    }

    return engine_ctx.cache.pipelines.render_splats;
}

export function RenderSplats(passEncoder, camera, splats, target)
{
    let pipeline = GetPipeline(target.view_format, target.msaa);
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, camera.bind_group);
    passEncoder.setBindGroup(1, splats.bind_group_render_splats);
    //passEncoder.setVertexBuffer(0, splats.dIndices);    
    passEncoder.setVertexBuffer(0, splats.dIndices1);    
    passEncoder.drawIndirect(splats.dDrawIndirect, 0);    

}