extern crate nalgebra as na;
use image::{Rgba, imageops};
use imageproc::{drawing, corners};
use std::time::SystemTime;
use image_processing::{Pyramid, find_features, orientation};

mod rbrief {
    use rand::{Rng};
    use rand::distributions::{Uniform};

    pub struct Point {
        pub x: i32,
        pub y: i32
    }

    pub struct PairPoint(pub Point, pub Point);

    pub struct TestSet {
        pub set: Vec<PairPoint>
    }

    impl TestSet {
        pub fn new() -> TestSet {
            // 128 pairs of points in range -15 to +15
            let mut set = Vec::<PairPoint>::new();
            let mut rng = rand::thread_rng();
            let d = Uniform::new_inclusive(-15, 15);
            let v: Vec<i32> = (&mut rng).sample_iter(d).take(128 * 4).collect(); 
            for i in 0..128 {
                set.push(PairPoint(
                        Point {
                            x:v[i * 4],
                            y:v[i * 4 + 1]
                         },
                         Point {
                            x:v[i * 4 + 2],
                            y:v[i * 4 + 3]
                          }));
            }
            TestSet {
                set: set
            }
        }
    }

    pub fn rotate(set:&TestSet, angle:f32) -> TestSet {
        let c = f32::cos(angle);
        let s = f32::sin(angle);
        TestSet {
            set: set.set.iter().map(|PairPoint(p1, p2)|
                PairPoint(Point {
                        x:(c * p1.x as f32 - s * p1.y as f32) as i32,
                        y:(s * p1.x as f32 + c * p1.y as f32) as i32,
                     },
                     Point {
                        x:(c * p2.x as f32 - s * p2.y as f32) as i32,
                        y:(s * p2.x as f32 + c * p2.y as f32) as i32,
                      }))
            .collect()
        }
    }

    pub struct RBrief {
        sets: Vec<TestSet>,
        angle_per_set: f32
    }

    impl RBrief {
        fn new() -> RBrief {
            let mut sets = Vec::<TestSet>::new();
            sets.push(TestSet::new());
            let alpha = std::f32::consts::PI / 30.0;
            for i in 1..30 {
                sets.push(rotate(&sets[0], i as f32 * alpha))
            }
            RBrief {
                sets: sets,
                angle_per_set: alpha
            }
        }
    }

    // calculate the 128 bit score for a given integral image and test_set
    // calculate said score for given integral image, angle, prerotated test_set
}
#[cfg(test)]
mod tests {
    use super::*;
    use super::rbrief::PairPoint;
    use more_asserts::*;

    #[test]
    fn test_rbrief_test_set() {
        let t = rbrief::TestSet::new();
        assert_eq!(t.set.len(), 128);
        for PairPoint(p1, p2) in t.set.iter() {
            assert_le!(p1.x, 15);
            assert_le!(p1.y, 15);
            assert_le!(p2.x, 15);
            assert_le!(p2.y, 15);
            assert_ge!(p1.x, -15);
            assert_ge!(p1.y, -15);
            assert_ge!(p2.x, -15);
            assert_ge!(p2.y, -15);
         }
    }

    #[test]
    fn test_rbrief_rotate() {
        let t = rbrief::TestSet::new();
        let tr = rbrief::rotate(&t, std::f32::consts::PI / 2.0);
        for i in 0..128 {
            assert_le!((t.set[i].0.x -  tr.set[i].0.y).abs(), 1);
            assert_le!((t.set[i].0.y - -tr.set[i].0.x).abs(), 1);
            assert_le!((t.set[i].1.x -  tr.set[i].1.y).abs(), 1);
            assert_le!((t.set[i].1.y - -tr.set[i].1.x).abs(), 1);
        }
    }
}


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
