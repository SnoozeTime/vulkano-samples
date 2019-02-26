#[macro_use]
extern crate vulkano;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::image::ImageUsage;
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::device::{Queue, Device, DeviceExtensions};
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, Subpass, RenderPassAbstract};
use vulkano::format::Format;
use vulkano::image::SwapchainImage;
use vulkano::image::attachment::AttachmentImage;
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain::{AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError};
use vulkano::swapchain;
use vulkano::sync::{GpuFuture, FlushError};
use vulkano::sync;

use vulkano_win::VkSurfaceBuild;

use winit::{EventsLoop, Window, WindowBuilder, Event, WindowEvent};

use std::sync::Arc;
use std::iter;

#[derive(Debug, Clone)]
struct Vertex { position: [f32; 3], color: [f32; 3], }
impl_vertex!(Vertex, position, color);


struct ObjectPicker {
    queue: Arc<Queue>,

    // Tells the GPU where to write the color
    render_pass: Arc<RenderPassAbstract + Send + Sync>,
    pipeline: Arc<GraphicsPipelineAbstract + Send + Sync>,

    // Two attachments -> color and depth
    framebuffer: Arc<FramebufferAbstract + Send + Sync>,

    // color attachment
    image: Arc<AttachmentImage>,

    // Will have the data from `image` copied to it.
    buf: Arc<CpuAccessibleBuffer<[u8]>>,

}

impl ObjectPicker {

    fn new(queue: Arc<Queue>, dimensions: [u32; 2]) -> Self {

        // Create the image to which we are going to render to. This
        // is not a swapchain image as we do not render to screen.
        let image_usage = ImageUsage {
            transfer_source: true, // This is necessary to copy to external buffer
            .. ImageUsage::none()
        };

        let image = AttachmentImage::with_usage(
            queue.device().clone(),
            dimensions,
            Format::R8G8B8A8Unorm, // simple format for encoding the ID as a color
            image_usage).unwrap();

        let depth_buffer = AttachmentImage::transient(
            queue.device().clone(),
            dimensions,
            Format::D16Unorm).unwrap();

        let render_pass = Arc::new(vulkano::single_pass_renderpass!(
                queue.device().clone(),
                attachments: {
                    color: {
                        load: Clear,
                        store: Store,
                        format: Format::R8G8B8A8Unorm,
                        samples: 1,
                    },
                    depth: {
                        load: Clear,
                        store: DontCare,
                        format: Format::D16Unorm,
                        samples: 1,
                    }
                },
                pass: {
                    color: [color],
                    depth_stencil: {depth}
                }
        ).unwrap());

        // Use our custom image in the framebuffer.
        let framebuffer = Arc::new(Framebuffer::start(render_pass.clone())
                                   .add(image.clone()).unwrap()
                                   .add(depth_buffer.clone()).unwrap()
                                   .build().unwrap());

        // That is the CPU accessible buffer to which we'll transfer the image content
        // so that we can read the data. It should be as large as 4 the number of pixels (because we
        // store rgba value, so 4 time u8)
        let buf = CpuAccessibleBuffer::from_iter(
            queue.device().clone(), BufferUsage::all(),
            (0 .. dimensions[0] * dimensions[1] * 4).map(|_| 0u8)).expect("Failed to create buffer");


        //
        let vs = pick_vs::Shader::load(queue.device().clone()).unwrap();
        let fs = pick_fs::Shader::load(queue.device().clone()).unwrap();
        let pipeline = Arc::new(GraphicsPipeline::start()
                                .vertex_input_single_buffer::<Vertex>()
                                .vertex_shader(vs.main_entry_point(), ())
                                .triangle_list()
                                .viewports_dynamic_scissors_irrelevant(1)
                                .depth_stencil_simple_depth()
                                .viewports(iter::once(Viewport {
                                    origin: [0.0, 0.0],
                                    dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                                    depth_range: 0.0 .. 1.0,
                                }))
                                .fragment_shader(fs.main_entry_point(), ())
                                .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                                .build(queue.device().clone())
                                .unwrap());

        ObjectPicker {
            queue,
            render_pass,
            pipeline,
            framebuffer,
            image,
            buf,
        }

    }

    fn create_pushconstants(id: usize) -> pick_vs::ty::PushConstants {
        pick_vs::ty::PushConstants {
            color: 
                [
                ((id & 0xFF) as f32) / 255.0,
                ((id >> 8) & 0xFF) as f32 / 255.0,
                ((id >> 16) & 0xFF) as f32 / 255.0,
                1.0], // Transparent means no entity.
        }
    }

    /// Return either ID of picked object or None if did not click on anything
    fn pick_object(&mut self, x: f64, y: f64, objects: Vec<Arc<CpuAccessibleBuffer<[Vertex]>>>) -> Option<usize> {

        let clear_values = vec!([0.0, 0.0, 0.0, 0.0].into(), 1f32.into());

        let mut command_buffer_builder = AutoCommandBufferBuilder::primary_one_time_submit(
            self.queue.device().clone(),
            self.queue.family()).unwrap()
            .begin_render_pass(self.framebuffer.clone(), false, clear_values).unwrap();

        // Now, render all objects and use the ID as push constant.
        for (id, object) in objects.iter().enumerate() {
                
            let push_constant = ObjectPicker::create_pushconstants(id);
            command_buffer_builder = command_buffer_builder.draw(
                self.pipeline.clone(),
                &DynamicState::none(),
                vec![object.clone()],
                (),
                push_constant,
                ).unwrap();
        }

        command_buffer_builder = command_buffer_builder.end_render_pass().unwrap();

        // Now copy the image to the CPU accessible buffer.
        command_buffer_builder = command_buffer_builder
            .copy_image_to_buffer(self.image.clone(), self.buf.clone()).unwrap();

        let command_buffer = command_buffer_builder.build().unwrap();

        

        None
    }

}


fn main() {
    let instance = {
        let extensions = vulkano_win::required_extensions();
        Instance::new(None, &extensions, None).unwrap()
    };

    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    println!("Using device: {} (type: {:?})", physical.name(), physical.ty());
    let mut events_loop = EventsLoop::new();
    let surface = WindowBuilder::new().build_vk_surface(&events_loop, instance.clone()).unwrap();
    let window = surface.window();

    let queue_family = physical.queue_families().find(|&q| {
        q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
    }).unwrap();
    let device_ext = DeviceExtensions { khr_swapchain: true, .. DeviceExtensions::none() };
    let (device, mut queues) = Device::new(physical, physical.supported_features(), &device_ext,
                                           [(queue_family, 0.5)].iter().cloned()).unwrap();

    let queue = queues.next().unwrap();
    let (mut swapchain, images) = {

        let caps = surface.capabilities(physical).unwrap();
        let usage = caps.supported_usage_flags;
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;

        let initial_dimensions = if let Some(dimensions) = window.get_inner_size() {
            let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
            [dimensions.0, dimensions.1]
        } else {
            return;
        };

        Swapchain::new(device.clone(), surface.clone(), caps.min_image_count, format,
        initial_dimensions, 1, usage, &queue, SurfaceTransform::Identity, alpha,
        PresentMode::Fifo, true, None).unwrap()
    };


    // MY THREE OBJECTS.
    // a triangle
    let vertex_buffer = {

        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), [
                                       Vertex { position: [-1.0, -0.25, 0.0], color: [0.0, 0.0, 1.0]},
                                       Vertex { position: [0.0, 0.5, 0.0], color: [0.0, 0.0, 1.0]},
                                       Vertex { position: [0.25, -0.1, 0.2], color: [0.0, 0.0, 1.0]}
        ].iter().cloned()).unwrap()
    };

    // A quad
    let vertex_buffer_2 = {
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), [
                                       Vertex { position: [-0.5, -0.5, 0.0], color: [1.0, 1.0, 0.0]},
                                       Vertex { position: [0.5, -0.5, 0.0], color: [1.0, 1.0, 0.0]},
                                       Vertex { position: [-0.5, 0.5, 0.0], color: [1.0, 1.0, 0.0]},

                                       Vertex { position: [0.5, -0.5, 0.0], color: [1.0, 1.0, 0.0]},
                                       Vertex { position: [0.5, 0.5, 0.0], color: [1.0, 1.0, 0.0]},
                                       Vertex { position: [-0.5, 0.5, 0.0], color: [1.0, 1.0, 0.0]}
        ].iter().cloned()).unwrap()

    };

    // A second triangle
    let vertex_buffer_3 = {
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), [
                                       Vertex { position: [0.0, 1.0, 0.5], color: [1.0, 0.0, 0.0]},
                                       Vertex { position: [1.0, -1.0, 0.5], color: [1.0, 0.0, 0.0]},
                                       Vertex { position: [1.0, 1.0, 0.5], color: [1.0, 0.0, 0.0]}
        ].iter().cloned()).unwrap()
    };

    let vs = vs::Shader::load(device.clone()).unwrap();
    let fs = fs::Shader::load(device.clone()).unwrap();


    let render_pass = Arc::new(single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.format(),
                    samples: 1,
                },
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: Format::D16Unorm,
                    samples: 1,

                }
            },

            pass: {
                color: [color],
                depth_stencil: {depth}
            }
    ).unwrap());

    let pipeline = Arc::new(GraphicsPipeline::start()
                            .vertex_input_single_buffer()
                            .vertex_shader(vs.main_entry_point(), ())
                            .triangle_list()
                            .viewports_dynamic_scissors_irrelevant(1)
                            .depth_stencil_simple_depth()
                            .fragment_shader(fs.main_entry_point(), ())
                            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                            .build(device.clone())
                            .unwrap());

    let mut dynamic_state = DynamicState { line_width: None, viewports: None, scissors: None };

    let mut framebuffers = window_size_dependent_setup(device.clone(), &images, render_pass.clone(), &mut dynamic_state);


    // -------------------------------------------
    // Now we redo the same but for object picking
    // --------------------------------------------
    let mut recreate_swapchain = false;
    let mut previous_frame_end = Box::new(sync::now(device.clone())) as Box<GpuFuture>;



    loop {
        previous_frame_end.cleanup_finished();

        if recreate_swapchain {
            let dimensions = if let Some(dimensions) = window.get_inner_size() {
                let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
                [dimensions.0, dimensions.1]
            } else {
                return;
            };
            let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimensions) {
                Ok(r) => r,
                Err(SwapchainCreationError::UnsupportedDimensions) => continue,
                Err(err) => panic!("{:?}", err)
            };

            swapchain = new_swapchain;
            framebuffers = window_size_dependent_setup(device.clone(), &new_images, render_pass.clone(), &mut dynamic_state);
            recreate_swapchain = false;
        }

        let (image_num, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(), None) {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                recreate_swapchain = true;
                continue;
            },
            Err(err) => panic!("{:?}", err)
        };

        // Specify the color to clear the framebuffer with i.e. blue
        let clear_values = vec!([0.0, 0.0, 0.0, 1.0].into(), 1f32.into());

        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
            .begin_render_pass(framebuffers[image_num].clone(), false, clear_values)
            .unwrap()
            .draw(pipeline.clone(), &dynamic_state, vertex_buffer.clone(), (), ())
            .unwrap()
            .draw(pipeline.clone(), &dynamic_state, vertex_buffer_2.clone(), (), ()).unwrap()
            .draw(pipeline.clone(), &dynamic_state, vertex_buffer_3.clone(), (), ()).unwrap()
            .end_render_pass()
            .unwrap()
            // Finish building the command buffer by calling `build`.
            .build().unwrap();

        let future = previous_frame_end.join(acquire_future)
            .then_execute(queue.clone(), command_buffer).unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();


        match future {
            Ok(future) => {
                previous_frame_end = Box::new(future) as Box<_>;
            }
            Err(FlushError::OutOfDate) => {
                recreate_swapchain = true;
                previous_frame_end = Box::new(sync::now(device.clone())) as Box<_>;
            }
            Err(e) => {
                println!("{:?}", e);
                previous_frame_end = Box::new(sync::now(device.clone())) as Box<_>;
            }
        }

        let mut done = false;
        events_loop.poll_events(|ev| {
            match ev {
                Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => done = true,
                Event::WindowEvent { event: WindowEvent::Resized(_), .. } => recreate_swapchain = true,
                _ => ()
            }
        });
        if done { return; }
    }
}

fn window_size_dependent_setup(
    device: Arc<Device>,
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState
) -> Vec<Arc<FramebufferAbstract + Send + Sync>> {

    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0 .. 1.0,
    };

    dynamic_state.viewports = Some(vec!(viewport));

    images.iter().map(|image| {
        let depth_buffer = AttachmentImage::transient(device.clone(), dimensions, Format::D16Unorm).unwrap();
        Arc::new(
            Framebuffer::start(render_pass.clone())
            .add(image.clone()).unwrap()
            .add(depth_buffer.clone()).unwrap()
            .build().unwrap()
        ) as Arc<FramebufferAbstract + Send + Sync>
    }).collect::<Vec<_>>()
}

// The next step is to create the shaders.
mod vs {
    vulkano_shaders::shader!{
        ty: "vertex",
        src: "
#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 0) out vec3 frag_color;

void main() {
    gl_Position = vec4(position, 1.0);
    frag_color = color;
}"
}
}

mod fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        src: "
#version 450

layout(location = 0) in vec3 frag_color;
layout(location = 0) out vec4 f_color;

void main() {
    f_color = vec4(frag_color, 1.0);
}
"
}
}

mod pick_vs {
    vulkano_shaders::shader!{
        ty: "vertex",
        src: "
#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 0) out vec4 frag_color;

layout (push_constant) uniform PushConstants {
        vec4 color;
} pushConstants;

void main() {
    gl_Position = vec4(position, 1.0);
    frag_color = pushConstants.color;
}"
}
}

mod pick_fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        src: "
#version 450

layout(location = 0) in vec4 frag_color;
layout(location = 0) out vec4 f_color;

void main() {
    f_color = frag_color;
}
"
}
}


