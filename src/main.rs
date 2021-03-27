extern crate nalgebra as na;
use image::{/*DynamicImage*/ Rgba, /*RgbaImage,*/ GrayImage, Luma, imageops /*, GenericImageView*/};
use imageproc::{drawing, corners, gradients};
//use rand::{thread_rng, Rng};
//use rand::distributions::{Uniform};
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

fn circular_window(r:f32) ->Vec<(i32, i32)> {
    let mut offsets = Vec::<(i32, i32)>::new();
    for y in (-r as i32)..((r + 1.0) as i32) {
        let x_max = (r * r - (y * y) as f32).sqrt();
        for x in (-x_max as i32)..((x_max + 1.0) as i32) {
            offsets.push((x, y));
        }
    }
    offsets
}

fn harris_score(src:&GrayImage, x:u32, y:u32, r:u32) -> f32 {
    let (w, h) = src.dimensions();
    if x < r || y < r || x > w || w - x < r || y > h || h - y < r {
        return 0.0;
    }
    let view = imageops::crop_imm(src, x - r, y - r, 2 * r + 1, 2 * r + 1).to_image();
    let view_ix = gradients::horizontal_sobel(&view);
    let view_iy = gradients::vertical_sobel(&view);
    let mut a = na::Matrix2::new(0.0, 0.0, 0.0, 0.0);
    let window = circular_window(r as f32 + 0.5);
    let s = 1.0 / ( window.len().pow(4) as f32 );
    let ri = r as i32;
    for (i, j) in window.iter() {
        let Luma([ix]) = view_ix.get_pixel((i + ri) as u32, (j + ri) as u32);
        let Luma([iy]) = view_iy.get_pixel((i + ri) as u32, (j + ri) as u32);
        let ix = *ix as f32;
        let iy = *iy as f32;
        a[(0, 0)] += ix * ix * s;
        a[(0, 1)] += ix * iy * s;
        a[(1, 0)] += ix * iy * s;
        a[(1, 1)] += iy * iy * s;
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
        corner.score = harris_score(src, x, y, 3);
    }

    // sort by score and pick the top half
    corners.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
    corners.truncate(corners.len() / 2);
    println!("sorted and truncated to {}", corners.len());
    corners
}

fn orientation(image:&GrayImage, x:u32, y:u32, r:u32) -> f32 {
    if x < r || y < r || x > image.width() - r || y > image.height() -r {
        return 0.0;
    }
    // mpq = sum((x^p)*(y^q)*I(x,y))
    // theta = atan2(m01, m10)
    let mut m01:f32 = 0.0;
    let mut m10:f32 = 0.0;
    let window = circular_window(r as f32 + 0.5);
    let xi = x as i32;
    let yi = y as i32;
    for (i, j) in window.iter() {
        let Luma([p]) = image.get_pixel((xi + i) as u32, (yi + j) as u32);
        m01 += (*j as f32) * (*p as f32);
        m10 += (*i as f32) * (*p as f32);
    }
    m01.atan2(m10)
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use image::{imageops, ImageBuffer, Luma};
    use imageproc::{geometric_transformations};
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
        let score = harris_score(&image, 4, 4, 3);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_harris_bounds() {
        let image = ImageBuffer::from_pixel(8, 8, Luma([128u8]));
        let score = harris_score(&image, 2, 2, 3);
        assert_eq!(score, 0.0);
        let score = harris_score(&image, 7, 9, 3);
        assert_eq!(score, 0.0);
    }
 
    #[test]
    fn test_harris_constant_gradient() {
        let mut image = GrayImage::new(20, 20);
        imageops::horizontal_gradient(&mut image, &Luma([0]), &Luma([255]));
        let score = harris_score(&image, 10, 10, 3);
        assert_le!(score, 0.0);
        // invert the gradient
        imageops::horizontal_gradient(&mut image, &Luma([255]), &Luma([0]));
        let score = harris_score(&image, 10, 10, 3);
        assert_le!(score, 0.0);
        // rotate by 90
        image = imageops::rotate90(&image);
        let score = harris_score(&image, 10, 10, 3);
        assert_le!(score, 0.0);
    }

    #[test]
    fn test_harris_corner() {
        let mut white = ImageBuffer::from_pixel(8, 8, Luma([255u8]));
        let black = ImageBuffer::from_pixel(4, 4, Luma([0u8]));
        imageops::replace(&mut white, &black, 0, 0);
        let score = harris_score(&white, 4, 4, 3);
        assert_gt!(score, 0.0);
        let white90 = imageops::rotate90(&white);
        let score = harris_score(&white90, 4, 4, 3);
        assert_gt!(score, 0.0);
    }

    #[test]
    fn test_harris_corner_invert() {
        let mut black = ImageBuffer::from_pixel(8, 8, Luma([0u8]));
        let white = ImageBuffer::from_pixel(4, 4, Luma([255u8]));
        imageops::replace(&mut black, &white, 0, 0);
        let score = harris_score(&black, 4, 4, 3);
        assert_gt!(score, 0.0);
        let black270 = imageops::rotate270(&black);
        let score = harris_score(&black270, 4, 4, 3);
        assert_gt!(score, 0.0);
    }

    #[test]
    fn test_circular_window() {
        let w0 = [(0, 0)];
        assert_eq!(&w0[..], &circular_window(0.0)[..]);
        let w25 = [(-1, -2), (0, -2), (1, -2),
            (-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1),
            (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
            (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1),
            (-1, 2), (0, 2), (1, 2)];
        assert_eq!(&w25[..], &circular_window(2.5)[..]);
    }

    #[test]
    fn test_orientation() {
        let mut image = GrayImage::new(20, 20);
        imageops::horizontal_gradient(&mut image, &Luma([0]), &Luma([255]));
        let angle = orientation(&image, 10, 10, 3);
        assert_eq!(angle, 0.0);
        let pi_over_4 = std::f32::consts::PI / 4.0;
        let im_r = geometric_transformations::rotate_about_center(&image,
                                pi_over_4,
                                geometric_transformations::Interpolation::Nearest,
                                Luma([0]));
        let angle = orientation(&im_r, 10, 10, 3);
        assert_eq!(angle, pi_over_4);
        let im_r = geometric_transformations::rotate_about_center(&image,
                                3.0 * pi_over_4,
                                geometric_transformations::Interpolation::Nearest,
                                Luma([0]));
        let angle = orientation(&im_r, 10, 10, 3);
        assert_eq!(angle, 3.0 * pi_over_4);
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
