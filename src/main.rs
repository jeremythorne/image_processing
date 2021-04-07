extern crate nalgebra as na;
use image::{Rgba, GrayImage, Luma, RgbaImage, imageops};
use imageproc::{drawing, corners, geometric_transformations};
//use std::time::SystemTime;
use std::collections::HashMap;
use itertools::Itertools;
mod rbrief;
use image_processing::{Pyramid, find_features, orientation};
use hamming_lsh;

struct Config {
    num_features: usize,
    fast_threshold: u8,
    num_pyramid_levels: u32,
    rbrief_test_set: rbrief::RBrief,
    lsh_k_l: (u32, u32),
    lsh_max_distance: u32
}

impl Default for Config {
    fn default() -> Config {
        Config {
            num_features: 500,
            fast_threshold: 32,
            num_pyramid_levels: 4,
            rbrief_test_set: rbrief::RBrief::new(),
            lsh_k_l: (4, 10),
            lsh_max_distance: 15,
        }
    }
}

struct Corner {
    corner: corners::Corner,
    angle: f32,
    descriptor: Option<u128>,
    level: u32
}



fn find_features_in_pyramid(pyramid:&Pyramid, config:&Config) -> Vec<Corner> {
    struct LevelCorner {
        corner: corners::Corner,
        level: u32
    }
    let mut level_corners = Vec::<LevelCorner>::new();
    
    for (i, image) in pyramid.images.iter().enumerate() {
        for f in find_features(image, config.fast_threshold).iter() {
            level_corners.push(
                LevelCorner {
                    level: i as u32,
                    corner: *f
                })
        }
    }

    // sort by score and pick the top num_features
    level_corners.sort_by(|a, b| a.corner.score.partial_cmp(&b.corner.score).unwrap());
    if level_corners.len() > config.num_features {
        level_corners.truncate(config.num_features);
    }

    let tests = &config.rbrief_test_set;

    fn describe_corner(pyramid:&Pyramid, tests:&rbrief::RBrief, c:&LevelCorner) -> Corner {
        let image = &pyramid.images[c.level as usize];
        let angle = orientation(image, c.corner.x, c.corner.y, 3);
        let descriptor = tests.describe(image, c.corner.x, c.corner.y, angle);
        Corner {
            corner: c.corner,
            angle: angle,
            descriptor: descriptor,
            level: c.level
        }
    }

    level_corners.iter()
        .map(|c| describe_corner(pyramid, tests, c))
        .collect()
}

fn find_multiscale_features(image:&GrayImage, config:&Config) -> Vec<Corner> {
    let pyramid = Pyramid::new(&image, config.num_pyramid_levels);
    find_features_in_pyramid(&pyramid, config)
}

fn find_matches<'a>(a:&'a Vec<Corner>, b:&Vec<Corner>, config:&Config) -> Vec<Option<&'a Corner>> {
    let mut lsh = hamming_lsh::HammingLSH::new(
        config.lsh_k_l.0, config.lsh_k_l.1);
    
    for c in a.iter() {
        if let Some(descriptor) = c.descriptor {
            lsh.insert(descriptor, c);
        }
    }

    b.iter()
        .map(|c| if let Some(d) = c.descriptor { 
            lsh.get(d, Some(config.lsh_max_distance)) } else { None })
        .map(|m| if let Some(m) = m { Some(*m.1) } else { None })
        .collect()
}

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

fn match_stats(corners:&Vec<Corner>, matches:&Vec<Option<&Corner>>) {
    let mut distances = HashMap::<u32, u32>::new();
    for (corner, omatch) in corners.iter().zip(matches.iter()) {
        if let Some(m) = omatch {
            let a = corner.descriptor.unwrap();
            let b = m.descriptor.unwrap();
            let d = hamming_lsh::hamming_distance(a, b);
            let count = distances.entry(d / 5).or_insert(1);
            *count += 1;
        }
    }
    println!("histogram of hamming distances between matches and original features");
    for k in distances.keys().sorted() {
        println!("distance {} - {} : {}", k * 5, (k + 1) * 5, distances.get(k).unwrap());
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


    // find ORB corners in both and matches between the pair
    let config = Config::default();
    let corners = find_multiscale_features(&src_image, &config);
    let corners_r = find_multiscale_features(&im_r, &config);
    let matches = find_matches(&corners, &corners_r, &config);

    match_stats(&corners_r, &matches);

    let mut dst = src_image.expand_palette(&palette, None);
    draw_features(&mut dst, &corners);
    dst.save("features.png").expect("couldn't save");

    let mut dst = im_r.expand_palette(&palette, None);
    draw_matches(&mut dst, &corners_r, &matches);
    dst.save("rotate_matches.png").expect("couldn't save");

}
