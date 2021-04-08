extern crate nalgebra as na;
use image::{Rgba, Luma, RgbaImage, imageops};
use imageproc::{drawing, geometric_transformations};
//use std::time::SystemTime;
use std::collections::HashMap;
use itertools::Itertools;
use image_processing::{Config, Corner, find_multiscale_features, find_matches};

fn draw_features(image:&mut RgbaImage, corners:&Vec<Corner>) {
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

fn draw_matches(image:&mut RgbaImage, corners:&Vec<Corner>, matches:&Vec<Option<&Corner>>) {
    let blue = Rgba([0u8, 0u8, 255u8, 128u8]);
    let red = Rgba([255u8, 0u8, 0u8, 128u8]);
    for (corner, omatch) in corners.iter().zip(matches.iter()) {
        let s = 1 << corner.level;
        let p = ((corner.corner.x * s) as i32, (corner.corner.y * s) as i32);
        drawing::draw_hollow_circle_mut(image, p, 3, blue);
        if let Some(m) = omatch {
            let s = 1 << m.level;
            let pm = ((m.corner.x * s) as f32, (m.corner.y * s) as f32);
            let line_start = (p.0 as f32 , p.1 as f32);
            let line_end = pm;
            drawing::draw_line_segment_mut( image, line_start, line_end, red);
        }
    }
}

fn expected_location(w:u32, h:u32, theta:f32, x:u32, y:u32) -> (u32, u32) {
    let rot = na::Rotation2::new(-theta);
    let trans1 = na::Translation2::new(-(w as f32 / 2.0), -(h as f32 / 2.0));
    let trans2 = na::Translation2::new(w as f32 / 2.0, h as f32 / 2.0);
    let transform = trans2 * rot * trans1;
    let expected = transform * na::Point2::new(x as f32, y as f32);
    (expected.x.round() as u32, expected.y.round() as u32)
}

fn match_stats(corners:&Vec<Corner>, matches:&Vec<Option<&Corner>>,
               transform: (u32, u32, f32)) {
    let (w, h, theta) = transform;

    struct Stats {
        distances: Vec<u32>
    };
    
    let mut tp_distances = 
        Stats {
            distances: Vec::<u32>::new()
        };
    let mut fp_distances = 
        Stats {
            distances: Vec::<u32>::new()
        };

    for (corner, omatch) in corners.iter().zip(matches.iter()) {
        if let Some(m) = omatch {
            let a = corner.descriptor.unwrap();
            let b = m.descriptor.unwrap();
            let d = hamming_lsh::hamming_distance(a, b);
            let e = expected_location(w, h, theta, corner.corner.x, corner.corner.y);
            let true_positive = (e.0 as i32 - m.corner.x as i32).abs() < 2 
                             && (e.1 as i32 - m.corner.y as i32).abs() < 2;
            let stats = if true_positive { &mut tp_distances } else { &mut fp_distances };
            stats.distances.push(d);
        }
    }
    println!("histogram of hamming distances between matches and original features");
    println!("true positives:");
    calc(&tp_distances);
    println!("false positives:");
    calc(&fp_distances);

    fn calc(stats:&Stats) {
        let mut distances = HashMap::<u32, u32>::new();
        for d in stats.distances.iter() {
             let count = distances.entry(d / 5).or_insert(1);
            *count += 1;  
        }
        for k in distances.keys().sorted() {
            println!("distance {} - {} : {}", k * 5, (k + 1) * 5, distances.get(k).unwrap());
        }
        let mean = stats.distances.iter().fold(0.0, |s, d| s + *d as f32) / stats.distances.len() as f32;
        let var = stats.distances.iter().fold(0.0, |v, d| v + (*d as f32 - mean) * (*d as f32 - mean)) / stats.distances.len() as f32;
        let stddev = var.sqrt();
        println!("mean: {}, stddev: {}", mean, stddev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expected_location() {
        let w = 20;
        let h = 20;
        // no movement if theta = 0
        assert_eq!(expected_location(w, h, 0.0, 5, 5), (5, 5));
        // centre doesn't move
        assert_eq!(expected_location(w, h, 0.1, w / 2, h / 2), (10, 10));
        // pi / 2 => clockwise rotation about centre
        let theta = std::f32::consts::PI / 2.0;
        assert_eq!(expected_location(w, h, theta, 16, 12), (12, 4));
    }
}

fn main() {
    println!("Hello, world!");

    // make a grey -> RGB pallete
    let mut palette = [(0u8, 0u8, 0u8); 256];
    for i in 0..255 {
        let g = i as u8;
        palette[i] = (g, g, g);
    }

    // open the source image as greyscale
    let src_image =
        image::open("res/im0.png").expect("failed to open image")
            .into_luma8();

    // resize to ~ 640x480
    let (mut w, mut h) = src_image.dimensions();
    h = h * 640 / w;
    w = 640;
    let src_image = imageops::resize(&src_image,
        w, h, imageops::FilterType::CatmullRom);
    println!("image {}x{}", w, h);

    // make a synthetically rotated copy
    let theta = std::f32::consts::PI / 30.0;
    let im_r = geometric_transformations::rotate_about_center(&src_image,
                            theta,
                            geometric_transformations::Interpolation::Nearest,
                            Luma([0]));

    println!("finding features and matches in a copy rotated by {} degrees",
             theta * 180.0 / std::f32::consts::PI);

    // find ORB corners in both and matches between the pair
    let mut config = Config::default();
    config.lsh_max_distance = 128;
    
    let corners = find_multiscale_features(&src_image, &config);

    #[cfg(feature = "perf")]
    for _ in 0..100 {
        let corners_r = find_multiscale_features(&im_r, &config);
        let _matches = find_matches(&corners, &corners_r, &config);
    }

    let corners_r = find_multiscale_features(&im_r, &config);
    let matches = find_matches(&corners, &corners_r, &config);
 
    match_stats(&corners_r, &matches, (w, h, theta));

    let mut dst = src_image.expand_palette(&palette, None);
    draw_features(&mut dst, &corners);
    dst.save("features.png").expect("couldn't save");

    let mut dst = im_r.expand_palette(&palette, None);
    draw_matches(&mut dst, &corners_r, &matches);
    dst.save("rotate_matches.png").expect("couldn't save");

}
