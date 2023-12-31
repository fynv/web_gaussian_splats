import { EngineContext } from "./engine/EngineContext.js"
import { CanvasContext } from "./engine/CanvasContext.js"
import { GPURenderTarget } from "./engine/renderers/GPURenderTarget.js"
import { PerspectiveCameraEx } from "./engine/cameras/PerspectiveCameraEx.js"
import { OrbitControls } from "./engine/controls/OrbitControls.js"
import { Color } from "./engine/math/Color.js"

import { SplatLoader } from './SplatLoader.js';
import { Splats } from './Splats.js';
import { RenderSplats } from "./render_splats.js"
import { Vector3 } from "./engine/math/Vector3.js"

export async function test()
{
    const canvas = document.getElementById('gfx');
    canvas.style.cssText = "position:absolute; width: 100%; height: 100%;";  
    
    const engine_ctx = new EngineContext();
    const canvas_ctx = new CanvasContext(canvas);
    await canvas_ctx.initialize();

    let resized = false;
    const size_changed = ()=>{
        canvas.width = canvas.clientWidth;
        canvas.height = canvas.clientHeight;        
        resized = true;
    };
    
    let observer = new ResizeObserver(size_changed);
    observer.observe(canvas);

    const info = document.getElementById("info");    

    const downloadProgress = (percent, percentLabel) => {
        info.innerHTML = `Download percentage: ${percent}`;
    };

    let splat_loader = new SplatLoader();
    
    let splatBuffer = await splat_loader.loadFromURL("./assets/truck_high.splat", downloadProgress);

    let camera = new PerspectiveCameraEx();
    camera.up.set(0, -1, -.17);
    camera.position.set(-5, -1, -1); 

    let controls = new OrbitControls(camera, canvas);    
    controls.target.set(1, 1, 0); 
    controls.enableDamping = true; 

    /*let splatBuffer = await splat_loader.loadFromURL("./assets/garden_high.splat", downloadProgress);

    let camera = new PerspectiveCameraEx();
    camera.up.set(0, -1, -0.54);
    camera.position.set(-3.15634, -0.16946, -0.51552); 

    let controls = new OrbitControls(camera, canvas);    
    controls.target.set(1.52976, 2.27776, 1.65898); 
    controls.enableDamping = true; */

    /*let splatBuffer = await splat_loader.loadFromURL("./assets/stump_high.splat", downloadProgress);

    let camera = new PerspectiveCameraEx();
    camera.up.set(0, -1, -1.0);
    camera.position.set(-3.3816, 1.96931, -1.71890); 

    let controls = new OrbitControls(camera, canvas);    
    controls.target.set(0.60910, 1.42099, 2.02511); 
    controls.enableDamping = true; */

    let splats = new Splats(splatBuffer);
    let render_target = new GPURenderTarget(canvas_ctx, false); 
    
    let t = Date.now();
    let dt = 1000.0/60.0;
    let last_view_vec = null;
    let last_camera_pos = null;
    let last_camera_quat = null;

    let input_r_aperture = document.getElementById("r_aperture");
    let input_focal_z = document.getElementById("focal_z");

    input_r_aperture.addEventListener("input", ()=>
    {
        let r_aperture = input_r_aperture.value * 0.001;
        splats.set_r_aperture(r_aperture);

    });

    input_focal_z.addEventListener("input", ()=>
    {
        let focal_z = input_focal_z.value;
        splats.set_focal_z(focal_z);
    });


    const render = () =>
    {
        let t1 = Date.now();
        dt = 0.9*dt + 0.1*(t1- t);
        let fps = 1000.0/dt;        
        t= t1;
        info.innerHTML = `${fps.toFixed(2)} fps`;

        controls.update();
        if (resized)
        {
            camera.aspect = canvas.width/canvas.height;
            camera.updateProjectionMatrix();
            resized = false;
        }

        render_target.update();

        camera.updateMatrixWorld(false);
    	camera.updateConstant();
        
        let half_fov_y = 25.0 * Math.PI/180.0;
        let half_fov_x = Math.atan(Math.tan(half_fov_y) *  camera.aspect);
        let alpha = Math.max(half_fov_y, half_fov_x);                     

        let view_vec = new Vector3(camera.matrixWorld.elements[8],camera.matrixWorld.elements[9],camera.matrixWorld.elements[10]);
        let sort = true;
        let thresh_angle = Math.PI*0.333;
        if (last_view_vec!=null)
        {            
            let delta_angle = Math.acos(Math.min(view_vec.dot(last_view_vec),1.0));
            if (delta_angle + alpha<thresh_angle)
            {
                sort = false;
            }
        }

        if (sort)
        {   
            splats.sort(view_vec);
            last_view_vec = view_vec;
        }

        let cam_pos = camera.position.clone();
        let cam_quat = camera.quaternion.clone();
        let cull = true;

        if (!sort && last_camera_pos!=null)
        {
            let d_pos = cam_pos.clone();
            d_pos.sub(last_camera_pos);
            let diff_pos = d_pos.length();
            let diff_quat = cam_quat.angleTo(last_camera_quat);
            if (diff_pos<0.01 && diff_quat < 0.01)
            {
                cull = false;
            }            
        }

        if (cull)
        {         
            splats.frustumCull(camera);            
            last_camera_pos = cam_pos;
            last_camera_quat = cam_quat;
        }
        splats.updateConstant(camera, window.devicePixelRatio, { x: render_target.width, y: render_target.height });

        let commandEncoder = engine_ctx.device.createCommandEncoder();

        let clearColor = new Color(0.0, 0.0, 0.0);
        let colorAttachment =  {            
            view: render_target.view_video,
            clearValue: { r: clearColor.r, g: clearColor.g, b: clearColor.b, a: 1 },
            loadOp: 'clear',
            storeOp: 'store'
        };
        

        let renderPassDesc = {
            colorAttachments: [colorAttachment]            
        }; 

        let passEncoder = commandEncoder.beginRenderPass(renderPassDesc);

        passEncoder.setViewport(
            0,
            0,
            render_target.width,
            render_target.height,
            0,
            1
        );
    
        passEncoder.setScissorRect(
            0,
            0,
            render_target.width,
            render_target.height,
        );

        RenderSplats(passEncoder, camera, splats, render_target);

        passEncoder.end();

        let cmdBuf = commandEncoder.finish();
        engine_ctx.queue.submit([cmdBuf]);     
        
        requestAnimationFrame(render);

    }
    render();

}

