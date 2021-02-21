extern crate nalgebra as na;
use image::{/*DynamicImage*/ Rgba, /*RgbaImage,*/ GrayImage, Luma, imageops /*, GenericImageView*/};
use imageproc::{drawing, corners, gradients};
// use rand::Rng;
use num;
use std::time::SystemTime;

struct Pyramid {
    images: Vec::<GrayImage>
}

fn get_safe_from_image(image: &GrayImage, x:i32, y:i32) -> Luma<u8> {
    let sx = num::clamp(x, 0, image.width() as i32 - 1);
    let sy = num::clamp(y, 0, image.height() as i32 - 1);
    *image.get_pixel(sx as u32, sy as u32)
}

fn get_safe_from_vec(v: &Vec<u8>, i: i32) -> u8 {
    let si = num::clamp(i, 0, v.len() as i32 - 1);
    v[si as usize]
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
    let (w, h) = src.dimensions();
    if x < 3 || y < 3 || x > w || w - x < 3 || y > h || h - y < 3 {
        return 0.0;
    }
    let view = imageops::crop_imm(src, x - 3, y - 3, 7, 7).to_image();
    let view_ix = gradients::horizontal_sobel(&view);
    let view_iy = gradients::vertical_sobel(&view);
    let mut a = na::Matrix2::new(0.0, 0.0, 0.0, 0.0);
    let s = 1.0 / ( 49.0 * 49.0 );
    for j in 0..7 {
        for i in 0..7 {
            let Luma([ix]) = view_ix.get_pixel(i, j);
            let Luma([iy]) = view_iy.get_pixel(i, j);
            let ix = *ix as f32;
            let iy = *iy as f32;
            a[(0, 0)] += ix * ix * s;
            a[(0, 1)] += ix * iy * s;
            a[(1, 0)] += ix * iy * s;
            a[(1, 1)] += iy * iy * s;
        }
    }
    let score = a.determinant() - 0.06 * a.trace() * a.trace();
    score
}

fn find_features(src:&GrayImage) -> Vec<corners::Corner> {
    let threshold = 32;
    let mut corners = corners::corners_fast9(src, threshold);
    println!("threshold: {} corners: {}", threshold, corners.len());

    for mut corner in corners.iter_mut() {
        let x = corner.x;
        let y = corner.y;
        corner.score = harris_score(src, x, y);
    }

    // sort by score and pick the top half
    corners.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
    corners.truncate(corners.len() / 2);
    println!("sorted and truncated to {}", corners.len());
    corners
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{imageops, ImageBuffer, Luma};
    use more_asserts::*;

    #[test]
    fn test_pyramid() {
        let image = ImageBuffer::from_pixel(8, 8, Luma([128u8]));
        let pyramid = Pyramid::new(&image, 4);
        assert_eq!(pyramid.images.len(), 4);
        assert_eq!(pyramid.images[3].dimensions(), (1, 1));
        assert_eq!(*pyramid.images[3].get_pixel(0, 0), Luma([128u8]));
    }

    #[test]
    fn test_harris_flat() {
        let image = ImageBuffer::from_pixel(8, 8, Luma([128u8]));
        let score = harris_score(&image, 4, 4);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_harris_bounds() {
        let image = ImageBuffer::from_pixel(8, 8, Luma([128u8]));
        let score = harris_score(&image, 2, 2);
        assert_eq!(score, 0.0);
        let score = harris_score(&image, 7, 9);
        assert_eq!(score, 0.0);
    }
 
    #[test]
    fn test_harris_constant_gradient() {
        let mut image = GrayImage::new(20, 20);
        imageops::horizontal_gradient(&mut image, &Luma([0]), &Luma([255]));
        let score = harris_score(&image, 10, 10);
        assert_le!(score, 0.0);
        // invert the gradient
        imageops::horizontal_gradient(&mut image, &Luma([255]), &Luma([0]));
        let score = harris_score(&image, 10, 10);
        assert_le!(score, 0.0);
        // rotate by 90
        image = imageops::rotate90(&image);
        let score = harris_score(&image, 10, 10);
        assert_le!(score, 0.0);
    }

    #[test]
    fn test_harris_corner() {
        let mut white = ImageBuffer::from_pixel(8, 8, Luma([255u8]));
        let black = ImageBuffer::from_pixel(4, 4, Luma([0u8]));
        imageops::replace(&mut white, &black, 0, 0);
        let score = harris_score(&white, 4, 4);
        assert_gt!(score, 0.0);
        let white90 = imageops::rotate90(&white);
        let score = harris_score(&white90, 4, 4);
        assert_gt!(score, 0.0);
    }

    #[test]
    fn test_harris_corner_invert() {
        let mut black = ImageBuffer::from_pixel(8, 8, Luma([0u8]));
        let white = ImageBuffer::from_pixel(4, 4, Luma([255u8]));
        imageops::replace(&mut black, &white, 0, 0);
        let score = harris_score(&black, 4, 4);
        assert_gt!(score, 0.0);
        let black270 = imageops::rotate270(&black);
        let score = harris_score(&black270, 4, 4);
        assert_gt!(score, 0.0);
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

    let mut palette = [(0u8, 0u8, 0u8); 256];
    for i in 0..255 {
        let g = i as u8;
        palette[i] = (g, g, g);
    }
    let mut dst = src_image.expand_palette(&palette, None);
     
    for level in 0..pyramid.images.len() {
        let now = SystemTime::now();
        let corners = find_features(&pyramid.images[level]);
        println!("level {} find features  took {}ms", level, now.elapsed().unwrap().as_millis());
        for corner in corners.iter() {
            let s = 1 << level;
            let p = ((corner.x * s) as i32, (corner.y * s) as i32);
            drawing::draw_hollow_circle_mut(&mut dst, p, 3 * s as i32, Rgba([0u8, 0u8, 255u8, 255u8]));  
        }
    }
    dst.save("out.png").expect("couldn't save");
}
