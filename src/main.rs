use image::{/*DynamicImage, Rgba, RgbaImage,*/ GrayImage, Luma, /*imageops, GenericImageView*/};
// use imageproc::drawing;
// use rand::Rng;

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


fn main() {
    println!("Hello, world!");
    let src_image =
        image::open("res/im0.png").expect("failed to open image");
    let pyramid = Pyramid::new(&src_image.into_luma8(), 4);
    pyramid.images[3].save("out.png").expect("couldn't save");
}
