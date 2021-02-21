extern crate nalgebra as na;
use image::{/*DynamicImage, Rgba, RgbaImage,*/ GrayImage, Luma, imageops /*, GenericImageView*/};
use imageproc::{/*drawing, */ corners, gradients};
// use rand::Rng;
use std::time::SystemTime;

struct Pyramid {
    images: Vec::<GrayImage>
}

fn get_safe_from_image(image: &GrayImage, x:i32, y:i32) -> Luma<u8> {
    if x >= 0 && (x as u32) < image.width()
        && y >=0 && (y as u32) < image.height() {
        image.get_pixel(x as u32, y as u32).clone()
    } else {
        Luma([0])
    }
}

fn get_safe_from_vec(v: &Vec<u8>, i: i32) -> u8 {
    if i >= 0 && (i as usize) < v.len() {
        v[i as usize]
    } else {
        0
    }
}

impl Pyramid {
    fn new(src_image:&GrayImage, levels:u32) -> Pyramid {
        let mut images = Vec::<GrayImage>::new();
        images.push(src_image.clone());
        let (mut w, mut h) = src_image.dimensions();
        let mut src = src_image;
        for _i in 0..levels {
            w = w / 2;
            h = h / 2;
            if w == 0 || h == 0 {
                break;
            }
            // binomial
            let kernel = [(-2, 1), (-1, 4), (0, 6), (1, 4), (2, 1)];
            let divisor = 16;
            let mut dst = GrayImage::new(w, h);
            for y in 0..h {
                let sw = w * 2;
                let mut row = Vec::<u8>::new();
                for sx in 0..sw {
                    let mut sum:u32 = 0;
                    for (a, w) in kernel.iter() {
                        sum += (get_safe_from_image(src, sx as i32, (y as i32) * 2 + a)[0] as u32) * w;
                    }
                    row.push((sum / divisor) as u8);
                }
                for x in 0..w {
                    let mut sum:u32 = 0;
                    for (a, w) in kernel.iter() {
                        sum += (get_safe_from_vec(&row, (x as i32) *2 + a) as u32) * w;
                    }
                    dst.put_pixel(x, y, Luma([(sum / divisor) as u8]));
                 }
            }
            images.push(dst);
            src = &images[images.len() - 1];
        }

        Pyramid {
            images: images
        }
    }
}

fn harris_score(src:&GrayImage, x:u32, y:u32) -> f32 {
    let view = imageops::crop_imm(src, x - 3, y - 3, 7, 7).to_image();
    let view_ix = gradients::horizontal_sobel(&view);
    let view_iy = gradients::vertical_sobel(&view);
    let mut a = na::Matrix2::new(0.0, 0.0, 0.0, 0.0);
    let s = 1.0 / ( 49.0 * 49.0 );
    for j in 0..7 {
        for i in 0..7 {
            let Luma([ix]) = view_ix.get_pixel(i, j);
            let Luma([iy]) = view_iy.get_pixel(i, j);
            a[(0, 0)] += (ix * ix) as f32 * s;
            a[(0, 1)] += (ix * iy) as f32 * s;
            a[(1, 0)] += (ix * iy) as f32 * s;
            a[(1, 1)] += (iy * iy) as f32 * s;
        }
    }
    let score = a.determinant() - 0.06 * a.trace() * a.trace();
    score
}

fn find_features(src:&GrayImage) {
    let threshold = 32;
    let corners = corners::corners_fast9(src, threshold);
    println!("threshold: {} corners: {}", threshold, corners.len());

    for corner in corners {
        // assume FAST returns no corners within 3 pixels of edge
        let x = corner.x;
        let y = corner.y;
        let score = harris_score(src, x, y);
        println!("{}, {} : {} {}", x, y, score, corner.score);
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
    let now = SystemTime::now();
    let pyramid = Pyramid::new(&src_image, 4);
    println!("pyramid took {}ms", now.elapsed().unwrap().as_millis());
    
    let now = SystemTime::now();
    find_features(&pyramid.images[0]);
    println!("find features  took {}ms", now.elapsed().unwrap().as_millis());
    pyramid.images[3].save("out.png").expect("couldn't save");
}
