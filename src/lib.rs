extern crate nalgebra as na;
use image::{GrayImage, Luma, imageops};
use imageproc::{corners, gradients};
use num;
pub mod rbrief;
use hamming_lsh;

pub struct Pyramid {
    pub images: Vec::<GrayImage>
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
    pub fn new(src_image:&GrayImage, levels:u32) -> Pyramid {
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
                let y2 = (y as i32) * 2;
                if y2 - 2 < 0 || y2 + 2 > (h as i32) - 1 {
                    // top / bottom where kernel overlaps image edge
                    for sx in 0..sw {
                        let sum = kernel.iter()
                            .fold(0, |sum, (a, w)|
                                sum + (get_safe_from_image(src, sx as i32, y2 + a)[0] as u32) * w);
                        row.push((sum / divisor) as u8);
                    }
                } else {
                    for sx in 0..sw {
                        let sum = kernel.iter()
                            .fold(0, |sum, (a, w)|
                                sum + (src.get_pixel(sx as u32, (y2 + a) as u32)[0] as u32) * w);
                        row.push((sum / divisor) as u8);
                    }
                }

                // left / right where kernel overlaps vec ends
                let x = 0;
                let sum = kernel.iter()
                    .fold(0, |sum, (a, w)|
                        sum + (get_safe_from_vec(&row, (x as i32) *2 + a) as u32) * w);
                dst.put_pixel(x, y, Luma([(sum / divisor) as u8]));
                let x = w - 1;
                let sum = kernel.iter()
                    .fold(0, |sum, (a, w)|
                        sum + (get_safe_from_vec(&row, (x as i32) *2 + a) as u32) * w);
                dst.put_pixel(x, y, Luma([(sum / divisor) as u8]));

                for x in 1..(w - 1) {
                    let sum = kernel.iter()
                        .fold(0, |sum, (a, w)|
                            sum + (row[((x as i32) * 2 + a) as usize] as u32) * w);
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

pub fn find_features(src:&GrayImage, threshold:u8) -> Vec<corners::Corner> {
    let corners = corners::corners_fast9(src, threshold);

    corners.iter()
        .map(|c|
             corners::Corner {
                 x: c.x,
                 y: c.y,
                 score: harris_score(src, c.x, c.y, 3)
            })
    .collect()
}

pub fn orientation(image:&GrayImage, x:u32, y:u32, r:u32) -> f32 {
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

pub struct Config {
    pub num_features: usize,
    pub fast_threshold: u8,
    pub num_pyramid_levels: u32,
    pub rbrief_test_set: rbrief::RBrief,
    pub lsh_k_l: (u32, u32),
    pub lsh_max_distance: u32
}

impl Default for Config {
    fn default() -> Config {
        Config {
            num_features: 500,
            fast_threshold: 32,
            num_pyramid_levels: 4,
            rbrief_test_set: rbrief::RBrief::from_test_set(
                rbrief::TestSet::load("res/trained_test_set.json").unwrap()),
            lsh_k_l: (4, 10),
            lsh_max_distance: 15,
        }
    }
}

pub struct Corner {
    pub corner: corners::Corner,
    pub angle: f32,
    pub descriptor: Option<u128>,
    pub level: u32
}

struct LevelCorner {
    corner: corners::Corner,
    level: u32
}

fn find_features_in_pyramid(pyramid:&Pyramid, config:&Config) -> Vec<LevelCorner> {
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
    level_corners
}

fn find_and_describe_features_in_pyramid(pyramid:&Pyramid, config:&Config) -> Vec<Corner> {
    let level_corners = find_features_in_pyramid(pyramid, config);
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

pub fn find_multiscale_features(image:&GrayImage, config:&Config) -> Vec<Corner> {
    let pyramid = Pyramid::new(&image, config.num_pyramid_levels);
    find_and_describe_features_in_pyramid(&pyramid, config)
}

pub fn find_matches<'a>(a:&'a Vec<Corner>, b:&Vec<Corner>, config:&Config) -> Vec<Option<&'a Corner>> {
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

pub fn add_image_to_trainer(trainer:&mut rbrief::Trainer, image:&GrayImage, config:&Config) {
    let pyramid = Pyramid::new(&image, config.num_pyramid_levels);
    let level_corners = find_features_in_pyramid(&pyramid, config);
    for c in level_corners {
        let im = &pyramid.images[c.level as usize];
        let angle = orientation(im, c.corner.x, c.corner.y, 3);
        trainer.accumulate(im, c.corner.x, c.corner.y, angle);
    }
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

