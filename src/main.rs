extern crate nalgebra as na;
use image::{Rgba, imageops};
use imageproc::{drawing, corners};
//use rand::{thread_rng, Rng};
//use rand::distributions::{Uniform};
use std::time::SystemTime;
use image_processing::{Pyramid, find_features, orientation};
/*
mod rbrief {
    struct Point {
        x: i32,
        y: i32
    }

    fn test_set() -> Vec<(Point, Point)> {
        // 128 pairs of points in range -15 to +15
        let mut set = Vec::<(Point, Point)>::new();
        let mut rng = thread_rng();
        let d = Uniform::new_inclusive(-15, 15);
        let v: Vec<i32> = (&mut rng).sample_iter(d).take(128 * 4).collect(); 
        for i in 0..128 {
            set.push((Point {
                        x:v[i * 4],
                        y:v[i * 4 + 1]
                     },
                     Point {
                        x:v[i * 4 + 2],
                        y:v[i * 4 + 3]
                      }));
        }
    }

    fn rotate(set:Vec<(Point, Point)>, angle:f32) -> Vec<(Point, Point)> {
        let c = cos(angle);
        let s = sin(angle);
        set.iter().map(|(p1, p2)|
            (Point {
                        x:(c * p1.x as f32 - s * p1.y as f32) as i32,
                        y:(s * p1.y as f32 + c * p1.y as f32) as i32,
                     },
                     Point {
                        x:(c * p2.x as f32 - s * p2.y as f32) as i32,
                        y:(s * p2.y as f32 + c * p2.y as f32) as i32,
                      }))
            .collect()
    }

    //TODO collect prerotated set at pi/30 intervals
    // calculate the 128 bit score for a given integral image and test_set
    // calculate said score for given integral image, angle, prerotated test_set
}
*/

struct ScaledCorner {
    corner: corners::Corner,
    angle: f32,
    // descriptor: u128,
    level: u32
}

fn find_features_in_pyramid(pyramid:&Pyramid) -> Vec<ScaledCorner> {
    let mut corners = Vec::<ScaledCorner>::new();
    // let tests = rbrief::test_set();
    for (i, image) in pyramid.images.iter().enumerate() {
        let level_corners = find_features(image);
        for c in level_corners {
            let angle = orientation(image, c.x, c.y, 3);
            //let descriptor = rbrief::describe(image, c.x, c.y, angle, i as u32, tests);
            corners.push(
                ScaledCorner {
                    corner: c,
                    angle: angle,
                    // descriptor: descriptor,
                    level: i as u32
                });
        }
    }
    corners
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
    let now = SystemTime::now();
    let pyramid = Pyramid::new(&src_image, 4);
    println!("pyramid took {}ms", now.elapsed().unwrap().as_millis());

    let mut palette = [(0u8, 0u8, 0u8); 256];
    for i in 0..255 {
        let g = i as u8;
        palette[i] = (g, g, g);
    }
    let mut dst = src_image.expand_palette(&palette, None);


    let now = SystemTime::now();
    let corners = find_features_in_pyramid(&pyramid);
    println!("find features  took {}ms", now.elapsed().unwrap().as_millis());

    let blue = Rgba([0u8, 0u8, 255u8, 128u8]);
    let red = Rgba([255u8, 0u8, 0u8, 128u8]);
    for corner in corners.iter() {
        let s = 1 << corner.level;
        let p = ((corner.corner.x * s) as i32, (corner.corner.y * s) as i32);
        drawing::draw_hollow_circle_mut(&mut dst, p, 3 * s as i32, blue);  
        let ln = ((s as f32) * 3.0 * corner.angle.cos(), (s as f32) * 3.0 * corner.angle.sin());
        let line_start = (p.0 as f32 + ln.0, p.1 as f32 + ln.1);
        let line_end = (p.0 as f32 - ln.0, p.1 as f32 - ln.1);
        drawing::draw_line_segment_mut(&mut dst, line_start, line_end, red);
    }
    dst.save("out.png").expect("couldn't save");
}
