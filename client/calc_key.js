const workgroup_size = 64;

const shader_code = `
@group(0) @binding(0)
var<uniform> uDir: vec3f;

@group(0) @binding(1)
var uCentersColorsTexture: texture_2d<u32>;

@group(0) @binding(2)
var<storage, read_write> bPosOut : array<f32>;

fn getDataIdx(idx: u32, stride: i32, offset: i32, width: u32) -> vec2u
{    
    let d = idx * u32(stride) + u32(offset);
    return vec2(d%width, d/width);
}

@compute @workgroup_size(${workgroup_size},1,1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>)
{
    let idx = GlobalInvocationID.x;
    if (idx >= arrayLength(&bPosOut)) 
    {
        return;
    }

    let width_CC = textureDimensions(uCentersColorsTexture).x;
    let sampledCenterColor = textureLoad(uCentersColorsTexture, getDataIdx(idx, 1, 0, width_CC), 0);
    let splatCenter = bitcast<vec3f>(sampledCenterColor.yzw);
    bPosOut[idx] = dot(uDir, splatCenter);
}
`;


function GetPipeline()
{
    if (!("calc_key" in engine_ctx.cache.pipelines))
    {
        let shaderModule = engine_ctx.device.createShaderModule({ code: shader_code });
        let bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.calc_key];
        const pipelineLayoutDesc = { bindGroupLayouts };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);

        engine_ctx.cache.pipelines.calc_key = engine_ctx.device.createComputePipeline({
            layout,
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
    }
    return engine_ctx.cache.pipelines.calc_key;
}

export function CalcKey(commandEncoder, splats)
{
    let pipeline = GetPipeline();

    let num_particles= splats.splatBuffer.getSplatCount();
    let num_groups = Math.floor((num_particles + workgroup_size - 1)/workgroup_size);
    let bind_group = splats.bind_group_calc_key;

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bind_group);
    passEncoder.dispatchWorkgroups(num_groups, 1,1); 
    passEncoder.end();
}
