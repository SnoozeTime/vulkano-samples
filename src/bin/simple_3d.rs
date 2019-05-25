#[macro_use]
extern crate vulkano;

use std::path::PathBuf;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBuffer, DynamicState};
use vulkano::device::{Device, DeviceExtensions, Queue};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
use vulkano::image::attachment::AttachmentImage;
use vulkano::image::ImageUsage;
use vulkano::image::SwapchainImage;
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::swapchain;
use vulkano::swapchain::{
    AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError,
};
use vulkano::sync;
use vulkano::sync::{FlushError, GpuFuture};

use vulkano_win::VkSurfaceBuild;

use winit::{ElementState, Event, EventsLoop, MouseButton, Window, WindowBuilder, WindowEvent};

use std::iter;
use std::sync::Arc;
use tobj;

/*
 *  The vertex data that will be passed as input to
 *  the graphic pipeline.
 * */
#[derive(Debug, Copy, Clone)]
pub struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
    texcoords: [f32; 2],
    normals: [f32; 3],
}

impl Vertex {
    fn new(x: f32, y: f32, z: f32, tx: f32, ty: f32, nx: f32, ny: f32, nz: f32) -> Self {
        let position = [x, y, z];
        let texcoords = [tx, ty];
        let color = [1.0, 1.0, 0.0];
        let normals = [nx, ny, nz];
        Vertex {
            position,
            color,
            texcoords,
            normals,
        }
    }
}
vulkano::impl_vertex!(Vertex, position, color, texcoords, normals);

/*
 * Model that are loaded in GPU memory
 * */
#[derive(Debug)]
pub struct Model {
    pub vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    pub index_buffer: Arc<CpuAccessibleBuffer<[u32]>>,
}

impl Model {
    // Uses the tinyobj library to load mesh from obj file.
    pub fn load_from_obj(device: Arc<Device>, filepath: PathBuf) -> Result<Model, String> {
        let box_obj = tobj::load_obj(&filepath);
        let (mut models, _materials) = box_obj.unwrap();

        let mut indices = Vec::new();
        let mut vertices = Vec::new();

        for model in &mut models {
            let mesh = &mut model.mesh;
            indices.append(&mut mesh.indices);

            // Verify everything is consistent
            if mesh.positions.len() % 3 != 0 {
                return Err("Mesh position vector length is not a multiple of 3.".to_owned());
            }
            if mesh.texcoords.len() % 2 != 0 {
                return Err("Mesh texture vector length is not a multiple of 2.".to_owned());
            }

            if mesh.normals.len() % 3 != 0 {
                return Err("Normals vector length is not a multiple of 3.".to_owned());
            }

            if (mesh.positions.len() / 3) != (mesh.texcoords.len() / 2)
                || (mesh.positions.len() != mesh.normals.len())
            {
                return Err(format!(
                    "Number of positions ({}) does not correspond to number of texture coords ({})",
                    mesh.positions.len() / 3,
                    mesh.texcoords.len() / 2
                ));
            }

            for v in 0..mesh.positions.len() / 3 {
                vertices.push(Vertex::new(
                    mesh.positions[3 * v],
                    mesh.positions[3 * v + 1],
                    mesh.positions[3 * v + 2],
                    mesh.texcoords[2 * v],
                    1.0 - mesh.texcoords[2 * v + 1],
                    mesh.normals[3 * v],
                    mesh.normals[3 * v + 1],
                    mesh.normals[3 * v + 2],
                ));
            }
        }

        Self::load_from_vec(device, vertices, indices)
    }

    pub fn load_from_vec(
        device: Arc<Device>,
        vertices: Vec<Vertex>,
        indices: Vec<u32>,
    ) -> Result<Model, String> {
        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            vertices.iter().cloned(),
        )
        .map_err(|_| "Error".to_owned())?;

        let index_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            indices.iter().cloned(),
        )
        .map_err(|_| "error".to_owned())?;

        Ok(Model {
            vertex_buffer,
            index_buffer,
        })
    }
}

fn main() {
    let instance = {
        let extensions = vulkano_win::required_extensions();
        Instance::new(None, &extensions, None).unwrap()
    };

    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    println!(
        "Using device: {} (type: {:?})",
        physical.name(),
        physical.ty()
    );
    let mut events_loop = EventsLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&events_loop, instance.clone())
        .unwrap();
    let window = surface.window();

    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
        .unwrap();
    let device_ext = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };
    let (device, mut queues) = Device::new(
        physical,
        physical.supported_features(),
        &device_ext,
        [(queue_family, 0.5)].iter().cloned(),
    )
    .unwrap();

    let queue = queues.next().unwrap();
    let ((mut swapchain, images), dimensions) = {
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

        (
            Swapchain::new(
                device.clone(),
                surface.clone(),
                caps.min_image_count,
                format,
                initial_dimensions,
                1,
                usage,
                &queue,
                SurfaceTransform::Identity,
                alpha,
                PresentMode::Fifo,
                true,
                None,
            )
            .unwrap(),
            initial_dimensions,
        )
    };

    let objects: Vec<_> =
        vec![Model::load_from_obj(device.clone(), PathBuf::from("cube.obj")).unwrap()];
    let vs = vs::Shader::load(device.clone()).unwrap();
    let fs = fs::Shader::load(device.clone()).unwrap();

    let render_pass = Arc::new(
        single_pass_renderpass!(
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
        )
        .unwrap(),
    );

    let pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .depth_stencil_simple_depth()
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap(),
    );

    let mut dynamic_state = DynamicState {
        line_width: None,
        viewports: None,
        scissors: None,
    };

    let mut framebuffers = window_size_dependent_setup(
        device.clone(),
        &images,
        render_pass.clone(),
        &mut dynamic_state,
    );

    // -------------------------------------------
    // Now we redo the same but for object picking
    // --------------------------------------------
    let mut recreate_swapchain = false;
    let mut previous_frame_end = Box::new(sync::now(device.clone())) as Box<GpuFuture>;

    loop {
        previous_frame_end.cleanup_finished();

        if recreate_swapchain {
            let dimensions = if let Some(dimensions) = window.get_inner_size() {
                let dimensions: (u32, u32) =
                    dimensions.to_physical(window.get_hidpi_factor()).into();
                [dimensions.0, dimensions.1]
            } else {
                return;
            };
            let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimensions) {
                Ok(r) => r,
                Err(SwapchainCreationError::UnsupportedDimensions) => continue,
                Err(err) => panic!("{:?}", err),
            };

            swapchain = new_swapchain;
            framebuffers = window_size_dependent_setup(
                device.clone(),
                &new_images,
                render_pass.clone(),
                &mut dynamic_state,
            );
            recreate_swapchain = false;
        }

        let (image_num, acquire_future) =
            match swapchain::acquire_next_image(swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    recreate_swapchain = true;
                    continue;
                }
                Err(err) => panic!("{:?}", err),
            };

        // Specify the color to clear the framebuffer with i.e. blue
        let clear_values = vec![[0.0, 0.0, 0.0, 1.0].into(), 1f32.into()];

        let mut command_buffer_builder =
            AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family())
                .unwrap()
                .begin_render_pass(framebuffers[image_num].clone(), false, clear_values)
                .unwrap();
        for (i, obj) in objects.iter().enumerate() {
            command_buffer_builder = command_buffer_builder
                .draw_indexed(
                    pipeline.clone(),
                    &dynamic_state,
                    obj.vertex_buffer.clone(),
                    obj.index_buffer.clone(),
                    (),
                    (),
                )
                .unwrap();
        }
        let command_buffer = command_buffer_builder
            .end_render_pass()
            .unwrap()
            // Finish building the command buffer by calling `build`.
            .build()
            .unwrap();

        let future = previous_frame_end
            .join(acquire_future)
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
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
        events_loop.poll_events(|ev| match ev {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => done = true,
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => recreate_swapchain = true,
            _ => (),
        });
        if done {
            return;
        }
    }
}

fn window_size_dependent_setup(
    device: Arc<Device>,
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState,
) -> Vec<Arc<FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0..1.0,
    };

    dynamic_state.viewports = Some(vec![viewport]);

    images
        .iter()
        .map(|image| {
            let depth_buffer =
                AttachmentImage::transient(device.clone(), dimensions, Format::D16Unorm).unwrap();
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(image.clone())
                    .unwrap()
                    .add(depth_buffer.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            ) as Arc<FramebufferAbstract + Send + Sync>
        })
        .collect::<Vec<_>>()
}

// The next step is to create the shaders.
mod vs {
    vulkano_shaders::shader! {
    ty: "vertex",
        src: "
#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec2 texcoords;
layout(location = 3) in vec3 normals;

layout(location = 0) out vec3 frag_color;

void main() {
    gl_Position = vec4(position, 1.0);
    frag_color = color;
}"
    }
}

mod fs {
    vulkano_shaders::shader! {
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
