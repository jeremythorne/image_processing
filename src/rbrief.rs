use rand::{Rng};
use rand::distributions::{Uniform};
use image::{imageops, GrayImage, Luma};
use imageproc::integral_image;
use imageproc::definitions::Image;

type GrayIntegral = Image<Luma<u32>>;

#[derive(Clone, Debug, PartialEq)]
pub struct Point {
    pub x: i32,
    pub y: i32
}

pub fn sample(image:&GrayIntegral, offset:&Point, p:&Point) -> u32 {
    let l = (offset.x + p.x - 2) as u32;
    let r = (offset.x + p.x + 2) as u32;
    let t = (offset.y + p.y - 2) as u32;
    let b = (offset.y + p.y + 2) as u32;
    integral_image::sum_image_pixels(image, l, t, r, b)[0]
}

#[derive(Clone, Debug, PartialEq)]
pub struct PairPoint(pub Point, pub Point);

impl PairPoint {
    fn all_pairs() -> RBriefPairIter {
        PairPoint ( 
            Point { x: -MAX, y: -MAX },
            Point { x: -MAX + WINDOW, y: -MAX }
        ).iter()
    }

    fn from(a:i32, b:i32, c:i32, d:i32) -> PairPoint {
        PairPoint ( 
            Point { x: a, y: b },
            Point { x: c, y: d }
        )
    }

    fn overlaps(&self) -> bool {
        let w = WINDOW;
        (self.0.x - self.1.x).abs() < w && (self.0.y - self.1.y).abs() < w
    }

    fn valid(&self) -> bool {
        self.0.x.abs() <= MAX
            && self.0.y.abs() <= MAX
            && self.1.x.abs() <= MAX
            && self.1.y.abs() <= MAX
    }

    fn iter(&self) -> RBriefPairIter {
        RBriefPairIter {
            pair: (*self).clone() 
        }
    }
}

pub fn test(image:&GrayIntegral, offset:&Point, p:&PairPoint) -> bool {
    sample(image, offset, &p.0) > sample(image, offset, &p.1)
}

pub struct TestSet {
    pub set: Vec<PairPoint>
}

impl TestSet {
    pub fn new() -> TestSet {
        // 128 pairs of points in range -13 to +13
        let mut set = Vec::<PairPoint>::new();
        let mut rng = rand::thread_rng();
        let d = Uniform::new_inclusive(-MAX, MAX);
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

fn describe_with_testset(image:&GrayIntegral, p:&Point, set: &TestSet) -> u128 {
    let mut d = 0u128;
    for i in 0..128 {
        if test(image, p, &set.set[i]) {
            d |= 1 << i;
        }
    }
    d
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
    pub fn new() -> RBrief {
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

    pub fn describe(&self, image:&GrayImage, x:u32, y:u32, angle:f32) -> Option<u128> {
        let index = ((angle / self.angle_per_set + 0.5) as usize) % self.sets.len();
        let r = RADIUS;
        let (w, h) = image.dimensions();
        if x < r || y < r || x + r > w || y + r > h {
            return None;
        }
        let view = imageops::crop_imm(image, x - r, y - r, 2 * r + 1, 2 * r + 1).to_image();
        let integral = integral_image::integral_image::<_, u32>(&view);
        let d = describe_with_testset(&integral, &Point{x:r as i32, y:r as i32}, &self.sets[index]);
        Some(d)
    }
}



struct RBriefPairIter {
    pair: PairPoint
}

impl Iterator for RBriefPairIter {
    type Item = PairPoint;
    fn next(&mut self) -> Option<PairPoint> {
        if !self.pair.valid() {
            println!("end {:?}", self.pair);
            None
        } else {
            let ret = Some(self.pair.clone());
            loop {
                if self.pair.1.x < MAX {
                    self.pair.1.x += 1;
                } else if self.pair.1.y < MAX {
                    self.pair.1.y += 1;
                    self.pair.1.x = -MAX;
                } else if self.pair.0.x < MAX {
                    self.pair.0.x += 1;
                    self.pair.1.y = self.pair.0.y;
                    self.pair.1.x = self.pair.0.x;
                } else {
                    self.pair.0.y += 1;
                    self.pair.0.x = -MAX;
                    self.pair.1.y = self.pair.0.y;
                    self.pair.1.x = self.pair.0.x;
                }
                if !self.pair.overlaps() {
                    break;
                }
            }
            ret
        }
    }
}

// max extents that we will sample from
// sample points are +/- 15 so at a radius of 15 * sqrt(2)
// then at each point we sample a square -2 to + 2
pub const HWIDTH:u32 = 15;
pub const HWINDOW:u32 = 2;
pub const WINDOW:i32 = HWINDOW as i32 * 2 + 1;
pub const MAX:i32 = (HWIDTH - HWINDOW) as i32;
pub const RADIUS:u32 = (HWIDTH as f64 * std::f64::consts::SQRT_2) as u32 + HWINDOW;

struct BitVec {
    vec:Vec<u8>,
    bit:u8
}

impl BitVec {
    fn new() -> BitVec {
        BitVec {
            vec: vec![0u8],
            bit: 0
        }
    }

    fn push(&mut self, a:bool) {
        let v = if a { 1 } else { 0 };
        let i = self.vec.len() - 1;
        self.vec[i] |= v << self.bit;
        if self.bit < 7 {
            self.bit += 1;
        } else {
            self.vec.push(0u8);
            self.bit = 0;
        }
    }

    fn len(&self) -> usize {
        (self.vec.len() - 1) * 8 + self.bit as usize
    }

    fn mean(&self) -> f32 {
        hamming::weight(&self.vec) as f32 /
            self.len() as f32
    }

    fn correlation(&self, b:&BitVec) -> f32 {
        hamming::distance(&self.vec, &b.vec) as f32 /
            self.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::PairPoint;
    use more_asserts::*;
    use image::{ImageBuffer, Luma};
    use imageproc::integral_image;

    #[test]
    fn test_rbrief_test_set() {
        let t = TestSet::new();
        assert_eq!(t.set.len(), 128);
        for PairPoint(p1, p2) in t.set.iter() {
            assert_le!(p1.x, MAX);
            assert_le!(p1.y, MAX);
            assert_le!(p2.x, MAX);
            assert_le!(p2.y, MAX);
            assert_ge!(p1.x, -MAX);
            assert_ge!(p1.y, -MAX);
            assert_ge!(p2.x, -MAX);
            assert_ge!(p2.y, -MAX);
         }
    }

    #[test]
    fn test_rbrief_rotate() {
        let t = TestSet::new();
        let tr = rotate(&t, std::f32::consts::PI / 2.0);
        for i in 0..128 {
            assert_le!((t.set[i].0.x -  tr.set[i].0.y).abs(), 1);
            assert_le!((t.set[i].0.y - -tr.set[i].0.x).abs(), 1);
            assert_le!((t.set[i].1.x -  tr.set[i].1.y).abs(), 1);
            assert_le!((t.set[i].1.y - -tr.set[i].1.x).abs(), 1);
        }
    }

    #[test]
    fn test_rbrief_sample() {
        let r = RADIUS;
        let image = ImageBuffer::from_pixel(r * 2, r * 2, Luma([1u8]));
        let integral = integral_image::integral_image::<_, u32>(&image);
        assert_eq!(sample(&integral, 
                                  &Point{x: r as i32, y: r as i32},
                                  &Point{x: 0, y:0}), 25);
    }

    #[test]
    fn test_rbrief_test() {
        let r = RADIUS;
        let mut white = ImageBuffer::from_pixel(r * 4, r * 2, Luma([255u8]));
        let black = ImageBuffer::from_pixel(r * 2, r * 2, Luma([0u8]));
        imageops::replace(&mut white, &black, 0, 0);
        let integral = integral_image::integral_image::<_, u32>(&white);
        let pair = PairPoint(
            Point { x: r as i32, y: r as i32 },
            Point { x: 3 * r as i32, y: r as i32 });
        assert_eq!(test(&integral, &Point{x: 0, y: 0}, &pair), false);
        let pair = PairPoint(pair.1, pair.0);
        assert_eq!(test(&integral, &Point{x: 0, y: 0}, &pair), true);
    }

    #[test]
    fn test_rbrief_all_pairs() {
        let mut i = PairPoint::all_pairs();
        assert_eq!(i.next(), Some(PairPoint::from(-MAX, -MAX, -MAX + WINDOW, -MAX)));
        assert_eq!(i.next(), Some(PairPoint::from(-MAX, -MAX, -MAX + WINDOW + 1, -MAX)));
        let mut i = PairPoint::from(MAX, 12, MAX, MAX).iter();
        let _p = i.next();
        assert_eq!(i.next(), Some(PairPoint::from( -MAX, MAX, -MAX + WINDOW, MAX)));
        let mut i = PairPoint::from(MAX - WINDOW, MAX, MAX, MAX).iter();
        let _p = i.next();
        assert_eq!(i.next(), None);
        // ORB 2012 makes this 205590, but I think they are wrong
        // they appear to do -MAX to MAX non inclusive - which with a 5x5 sub window only extends
        // from -15 to 14 inclusive i.e. a 30x30 patch, not 31x31.
        assert_eq!(PairPoint::all_pairs().count(), 240856);
    }
    
    #[test]
    fn test_bit_vec() {
        let mut b = BitVec::new();
        b.push(true);
        assert_eq!(b.len(), 1);
        assert_eq!(b.mean(), 1.0);
        b.push(false);
        assert_eq!(b.mean(), 0.5);
        let mut b = BitVec::new();
        for _i in 0..8 {
            b.push(true);
            b.push(false);
        }
        assert_eq!(b.len(), 16);
        assert_eq!(b.mean(), 0.5);
        let mut c = BitVec::new();
        for _i in 0..8 {
            c.push(true);
            c.push(true);
        }
        assert_eq!(b.correlation(&c), 0.5);
     }
}


