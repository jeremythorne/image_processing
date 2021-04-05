extern crate nalgebra as na;
use image::{Rgba, GrayImage, RgbaImage, imageops};
use imageproc::{drawing, corners};
//use std::time::SystemTime;
mod rbrief;
use image_processing::{Pyramid, find_features, orientation};

struct ScaledCorner {
    corner: corners::Corner,
    angle: f32,
    descriptor: Option<u128>,
    level: u32
}

fn find_features_in_pyramid(pyramid:&Pyramid) -> Vec<ScaledCorner> {
    let mut corners = Vec::<ScaledCorner>::new();
    let tests = rbrief::RBrief::new();
    for (i, image) in pyramid.images.iter().enumerate() {
        let level_corners = find_features(image);
        for c in level_corners {
            let angle = orientation(image, c.x, c.y, 3);
            let descriptor = tests.describe(image, c.x, c.y, angle);
            corners.push(
                ScaledCorner {
                    corner: c,
                    angle: angle,
                    descriptor: descriptor,
                    level: i as u32
                });
        }
    }
    corners
}

fn find_multiscale_features(image:&GrayImage) -> Vec<ScaledCorner> {
    let pyramid = Pyramid::new(&image, 4);
    find_features_in_pyramid(&pyramid)
}

fn draw_features(image:&mut RgbaImage, corners:&Vec<ScaledCorner>) {
    let blue = Rgba([0u8, 0u8, 255u8, 128u8]);
    let red = Rgba([255u8, 0u8, 0u8, 128u8]);
    for corner in corners.iter() {
        let s = 1 << corner.level;
        let p = ((corner.corner.x * s) as i32, (corner.corner.y * s) as i32);
        drawing::draw_hollow_circle_mut(image, p, 3 * s as i32, blue);  
        let ln = ((s as f32) * 3.0 * corner.angle.cos(), (s as f32) * 3.0 * corner.angle.sin());
        let line_start = (p.0 as f32 + ln.0, p.1 as f32 + ln.1);
        let line_end = (p.0 as f32 - ln.0, p.1 as f32 - ln.1);
        drawing::draw_line_segment_mut( image, line_start, line_end, red);
    }
}

fn main() {
    println!("Hello, world!");
    let src_image =
        image::open("res/im0.png").expect("failed to open image")
            .into_luma8();
    let (mut w, mut h) = src_image.dimensions();
    h = h * 640 / w;
    w = 640;
    let src_image = imageops::resize(&src_image,
        w, h, imageops::FilterType::CatmullRom);
    println!("image {}x{}", w, h);

    let mut palette = [(0u8, 0u8, 0u8); 256];
    for i in 0..255 {
        let g = i as u8;
        palette[i] = (g, g, g);
    }

    let corners = find_multiscale_features(&src_image);

    let mut dst = src_image.expand_palette(&palette, None);
    draw_features(&mut dst, &corners);
    dst.save("out.png").expect("couldn't save");

}
