extern crate nalgebra as na;
use image::{Rgba, Luma, RgbaImage, imageops};
use imageproc::{drawing, geometric_transformations};
//use std::time::SystemTime;
//use std::collections::HashMap;
//use itertools::Itertools;
use std::env;
use std::ops::IndexMut;
use std::fs;
use std::io;
use std::path::PathBuf;
use image_processing::{Config, Corner, add_image_to_trainer, find_multiscale_features, find_matches};
use image_processing::rbrief;

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

    let mut tp_distances = Vec::<u32>::new();
    let mut fp_distances = Vec::<u32>::new();

    for (corner, omatch) in corners.iter().zip(matches.iter()) {
        if let Some(m) = omatch {
            let a = corner.descriptor.unwrap();
            let b = m.descriptor.unwrap();
            let d = hamming_lsh::hamming_distance(a, b);
            let e = expected_location(w, h, theta, corner.corner.x, corner.corner.y);
            let true_positive = (e.0 as i32 - m.corner.x as i32).abs() < 2 
                             && (e.1 as i32 - m.corner.y as i32).abs() < 2;
            let stats = if true_positive { &mut tp_distances } else { &mut fp_distances };
            stats.push(d);
        }
    }
    let mut histogram = vec![(0, 0); 128];
    println!("true positives:");
    calc(&tp_distances, &mut histogram, true);
    println!("false positives:");
    calc(&fp_distances, &mut histogram, false);
    println!("features: {}, matches: {}",
             corners.len(), tp_distances.len() + fp_distances.len());
 
    let cumulative:Vec<(u32, u32)> =
                        histogram.iter()
                        .scan((0, 0), |s, (a, b)| {
                              s.0 += a;
                              s.1 += b;
                              Some(*s)})
                        .collect();

    let score:Vec<i32> = cumulative.iter()
                    .map(|(tp, fp)| (*tp as i32) - (*fp as i32))
                    .collect();

    let threshold = score.iter()
                    .enumerate()
                    .max_by_key(|(_, &value)| value)
                    .map(|(idx, _)| idx)
                    .unwrap();

    println!("an hamming distance threshold of {} would maximise true - false positives",
             threshold);

    fn calc(distances:&Vec<u32>, histogram:&mut Vec<(u32, u32)>, i:bool) {
        for d in distances.iter() {
            let mut count = histogram.index_mut(*d as usize);
            if i { count.0 += 1; } else { count.1 += 1};
        }
        let mean = distances.iter().fold(0.0, |s, d| s + *d as f32) / distances.len() as f32;
        let var = distances.iter().fold(0.0, |v, d| v + (*d as f32 - mean) * (*d as f32 - mean)) / distances.len() as f32;
        let stddev = var.sqrt();
        println!("count: {}, mean: {}, stddev: {}", distances.len(), mean, stddev);
    }
}

fn files(dir: &str) -> Result<Vec<PathBuf>, io::Error> {
    Ok(fs::read_dir(dir)?
        .into_iter()
        .filter(|r| r.is_ok()) // Get rid of Err variants for Result<DirEntry>
        .map(|r| r.unwrap().path()) // This is safe, since we only have the Ok variants
        .filter(|r| r.is_file()) // Filter out non-files
        .collect())
}

fn train_rbrief(dir_name:&str, count:usize) {
    // accumulate a bit array for every pair in 31x31 rect
    // for each image
    //   find features
    //   for every feature
    //     make integral image around point
    //     for every pair in 31x31
    //       rbrief test
    //       push result into 1 bit of u8 array
    // then perform greedy algorithm described in paper where 
    // mean = hamming::weight(t) / num_images
    // correlation = sum_over_R(hamming::distance(Ri, t))

    println!("training rBrief descriptor test set");
    let config = Config::default();
    let mut trainer = rbrief::Trainer::new();

    println!("using {} images from {}", count, dir_name);

    let mut num = 0;
    if let Ok(entries) = files(dir_name) {
        println!("{} files in folder", entries.len());
        for path in entries.iter() {
            println!("reading {:?}", path);
            
            // open the source image as greyscale
            if let Ok(src_image) = image::open(path) {
                let src_image = src_image.into_luma8();

                // resize to ~ 640x480
                let (mut w, mut h) = src_image.dimensions();
                h = h * 640 / w;
                w = 640;
                let image = imageops::resize(&src_image,
                    w, h, imageops::FilterType::CatmullRom);
                println!("adding to trainer");
                add_image_to_trainer(&mut trainer, &image, &config);
                num += 1;
                if num == count {
                    break;
                }
            }
        }
    } else {
        println!("couldn't read {}", dir_name);
        return;
    }
    if num == 0 {
        println!("didn't find any images");
        return;
    }
    trainer.make_test_set().save("trained_test_set.json").expect("failed to save trained set");
}

fn main() {
    println!("Hello, world!");

    let args:Vec<String> = env::args().collect();
    if args.len() == 4 && args[1] == "train" {
        train_rbrief(&args[2], args[3].parse().unwrap_or(1));
    }

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
    let theta = std::f32::consts::PI / 3.0;
    let im_r = geometric_transformations::rotate_about_center(&src_image,
                            theta,
                            geometric_transformations::Interpolation::Nearest,
                            Luma([0]));

    println!("finding features and matches in a copy rotated by {} degrees",
             theta * 180.0 / std::f32::consts::PI);

    // find ORB corners in both and matches between the pair
    let mut config = Config::default();
    config.lsh_k_l = (0, 1);
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
    
    config.lsh_max_distance = 15;
    let matches = find_matches(&corners, &corners_r, &config);

    let mut dst = im_r.expand_palette(&palette, None);
    draw_matches(&mut dst, &corners_r, &matches);
    dst.save("rotate_matches.png").expect("couldn't save");

}
