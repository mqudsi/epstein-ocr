#!/bin/sh
#![allow(missing_abi)] /*
# vim: set ft=rust :
# rust self-compiler by M. Al-Qudsi, licensed as public domain or MIT.
# See <https://neosmart.net/blog/self-compiling-rust-code/> for info & updates.
OUT=/tmp/$(printf "%s" $(realpath $(which "$0")) | md5sum | cut -d' '  -f1)
MD5=$(md5sum "$0" | cut -d' '  -f1)
(test -x "${OUT}" -a "${MD5}" = "$(cat "${OUT}.md5" 2>/dev/null)" ||
(grep -Eq '^\s*(\[([^][]*)])*\s*fn\s+main\b' "$0" && (rm -f ${OUT};
rustc "$0" -o ${OUT} && printf "%s" ${MD5} > ${OUT}.md5) || (rm -f ${OUT};
printf "fn main() {//%s\n}" "$(cat $0)" | rustc - -o ${OUT} &&
printf "%s" ${MD5} > ${OUT}.md5))) && exec ${OUT} "$@" || exit $? #*/

use std::env;
use std::fs::File;
use std::io;
use std::io::prelude::*;

#[repr(u8)]
#[allow(unused)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum JpegMarker {
    /// Byte stuffing (not a marker). Indicates a literal `0xFF` value.
    FF = 0x00,

    /// Temporary Private Use (Arithmetic Coding).
    /// Used only within arithmetic coding bitstreams; does not have a length field.
    TEM = 0x01,

    /// Start of Frame 0 (Baseline DCT).
    /// Huffman coding.
    SOF0 = 0xC0,

    /// Start of Frame 1 (Extended Sequential DCT).
    /// Huffman coding.
    SOF1 = 0xC1,

    /// Start of Frame 2 (Progressive DCT).
    /// Huffman coding.
    SOF2 = 0xC2,

    /// Start of Frame 3 (Lossless Sequential).
    /// Huffman coding.
    SOF3 = 0xC3,

    /// Define Huffman Table(s).
    /// Replaces the `SOF4` position in the sequence (0xC4).
    DHT = 0xC4,

    /// Start of Frame 5 (Differential Sequential DCT).
    /// Huffman coding.
    SOF5 = 0xC5,

    /// Start of Frame 6 (Differential Progressive DCT).
    /// Huffman coding.
    SOF6 = 0xC6,

    /// Start of Frame 7 (Differential Lossless Sequential).
    /// Huffman coding.
    SOF7 = 0xC7,

    /// Reserved for JPEG Extensions.
    /// Replaces the `SOF8` position in the sequence (0xC8).
    JPG = 0xC8,

    /// Start of Frame 9 (Extended Sequential DCT).
    /// Arithmetic coding.
    SOF9 = 0xC9,

    /// Start of Frame 10 (Progressive DCT).
    /// Arithmetic coding.
    SOF10 = 0xCA,

    /// Start of Frame 11 (Lossless Sequential).
    /// Arithmetic coding.
    SOF11 = 0xCB,

    /// Define Arithmetic Coding Conditioning(s).
    /// Replaces the `SOF12` position in the sequence (0xCC).
    DAC = 0xCC,

    /// Start of Frame 13 (Differential Sequential DCT).
    /// Arithmetic coding.
    SOF13 = 0xCD,

    /// Start of Frame 14 (Differential Progressive DCT).
    /// Arithmetic coding.
    SOF14 = 0xCE,

    /// Start of Frame 15 (Differential Lossless Sequential).
    /// Arithmetic coding.
    SOF15 = 0xCF,

    /// Restart with Modulo 8 Count 0.
    RST0 = 0xD0,
    /// Restart with Modulo 8 Count 1.
    RST1 = 0xD1,
    /// Restart with Modulo 8 Count 2.
    RST2 = 0xD2,
    /// Restart with Modulo 8 Count 3.
    RST3 = 0xD3,
    /// Restart with Modulo 8 Count 4.
    RST4 = 0xD4,
    /// Restart with Modulo 8 Count 5.
    RST5 = 0xD5,
    /// Restart with Modulo 8 Count 6.
    RST6 = 0xD6,
    /// Restart with Modulo 8 Count 7.
    RST7 = 0xD7,

    /// Start of Image.
    SOI = 0xD8,

    /// End of Image.
    EOI = 0xD9,

    /// Start of Scan (Transition to bitstream).
    SOS = 0xDA,

    /// Define Quantization Table(s).
    DQT = 0xDB,

    /// Define Number of Lines.
    /// Used if the image height was unknown at the start of the frame.
    DNL = 0xDC,

    /// Define Restart Interval.
    DRI = 0xDD,

    /// Define Hierarchical Progression.
    DHP = 0xDE,

    /// Expand Reference Component(s).
    EXP = 0xDF,

    /// Application Segment 0 (JFIF Header / AVI1).
    APP0 = 0xE0,
    /// Application Segment 1 (Exif Metadata / XMP).
    APP1 = 0xE1,
    /// Application Segment 2 (ICC Profile / FlashPix).
    APP2 = 0xE2,
    /// Application Segment 3 (Meta / JPS).
    APP3 = 0xE3,
    /// Application Segment 4 (Scalable Architecture).
    APP4 = 0xE4,
    /// Application Segment 5 (RMETA).
    APP5 = 0xE5,
    /// Application Segment 6 (NITF).
    APP6 = 0xE6,
    /// Application Segment 7 (Reserved).
    APP7 = 0xE7,
    /// Application Segment 8 (SPIFF).
    APP8 = 0xE8,
    /// Application Segment 9 (Media Jpeg).
    APP9 = 0xE9,
    /// Application Segment 10 (Photometadata).
    APP10 = 0xEA,
    /// Application Segment 11 (Jpeg-HDR).
    APP11 = 0xEB,
    /// Application Segment 12 (Picture Info / Photoshop Duck).
    APP12 = 0xEC,
    /// Application Segment 13 (Photoshop IRB / IPTC).
    APP13 = 0xED,
    /// Application Segment 14 (Adobe Color Profile).
    APP14 = 0xEE,
    /// Application Segment 15 (GraphicConverter / User defined).
    APP15 = 0xEF,

    /// JPEG Extension 0.
    JPG0 = 0xF0,
    /// JPEG Extension 1.
    JPG1 = 0xF1,
    /// JPEG Extension 2.
    JPG2 = 0xF2,
    /// JPEG Extension 3.
    JPG3 = 0xF3,
    /// JPEG Extension 4.
    JPG4 = 0xF4,
    /// JPEG Extension 5.
    JPG5 = 0xF5,
    /// JPEG Extension 6.
    JPG6 = 0xF6,
    /// JPEG Extension 7 (Reserved).
    JPG7 = 0xF7,
    /// JPEG Extension 8 (Reserved).
    JPG8 = 0xF8,
    /// JPEG Extension 9 (Reserved).
    JPG9 = 0xF9,
    /// JPEG Extension 10 (Reserved).
    JPG10 = 0xFA,
    /// JPEG Extension 11 (Reserved).
    JPG11 = 0xFB,
    /// JPEG Extension 12 (Reserved).
    JPG12 = 0xFC,
    /// JPEG Extension 13 (Reserved).
    JPG13 = 0xFD,

    /// Comment.
    COM = 0xFE,
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: jpeg_debug <file.jpg>");
        return Ok(());
    }

    let path = &args[1];
    let mut file = File::open(path)?;
    let metadata = file.metadata()?;
    let file_size = metadata.len();
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    println!("[*] Analyzing: {} ({} bytes)", path, file_size);

    let mut cursor: usize = 0;

    // Validation: check for SOI (Start of Image)
    if buffer.len() < 2 || buffer[0] != 0xFF || buffer[1] != JpegMarker::SOI as u8 {
        report_error(
            0,
            &buffer[0..2],
            "Missing SOI marker. File is either not a JPEG or the header is destroyed.",
        );
        return Ok(());
    }
    println!("0x{:08X} | Marker: FF D8 (SOI) | OK", 0);
    cursor += 2;

    // Parse TLV Segments until SOS
    loop {
        if cursor + 4 > buffer.len() {
            report_error(cursor, &[], "Unexpected EOF while reading segment headers.");
            break;
        }

        if buffer[cursor] != 0xFF {
            report_error(
                cursor,
                &[buffer[cursor]],
                "Expected marker prefix 0xFF but found data. Marker desynchronization occurred.",
            );
            return Ok(());
        }

        let marker = buffer[cursor + 1];

        // SOS is special: it marks the end of metadata and the start of the bitstream
        if marker == JpegMarker::SOS as u8 {
            let len = ((buffer[cursor + 2] as u16) << 8) | (buffer[cursor + 3] as u16);
            println!(
                "0x{:08X} | Marker: FF DA (SOS) | Length: {} | Start of Bitstream",
                cursor, len
            );
            cursor += len as usize + 2;
            break;
        }

        // Standard TLV (Type-Length-Value) Segment
        let len = ((buffer[cursor + 2] as u16) << 8) | (buffer[cursor + 3] as u16);
        println!(
            "0x{:08X} | Marker: FF {:02X} | Length: {:5} | Segment OK",
            cursor, marker, len
        );

        cursor += len as usize + 2;

        if cursor >= buffer.len() {
            report_error(cursor, &[], "File ended before SOS marker found.");
            return Ok(());
        }
    }

    // Scan the entropy-coded segment (the "bitstream").
    // We look for illegal FF markers. In the bitstream, FF must be followed by 00.
    // If it is followed by anything else (except Restart Markers), it's a technical corruption.
    println!("[*] Scanning bitstream for byte-stuffing integrity...");

    while cursor < buffer.len() - 1 {
        if buffer[cursor] == 0xFF {
            let next_byte = buffer[cursor + 1];
            match next_byte {
                0x00 => { /* Byte stuffed 0xFF, this is correct */ }
                0xD0..=0xD7 => { /* Restart Marker, valid in bitstream */ }
                0xD9 => {
                    println!(
                        "0x{:08X} | Marker: FF D9 (EOI) | End of Image reached.",
                        cursor
                    );
                    if cursor + 2 < buffer.len() {
                        println!(
                            "0x{:08X} | Warning: {} extraneous bytes found after EOI.",
                            cursor + 2,
                            buffer.len() - (cursor + 2)
                        );
                    }
                    return Ok(());
                }
                _ => {
                    report_error(cursor, &buffer[cursor..cursor+2],
                        &format!("Illegal marker 0xFF {:02X} inside bitstream. This violates byte-stuffing rules and will cause decoders to abort.", next_byte));
                    return Ok(());
                }
            }
        }
        cursor += 1;
    }

    report_error(
        buffer.len(),
        &[],
        "Missing EOI (FF D9) marker. The image is likely truncated.",
    );
    Ok(())
}

fn report_error(offset: usize, bytes: &[u8], msg: &str) {
    let hex_found = bytes
        .iter()
        .map(|b| format!("{:02X}", b))
        .collect::<Vec<String>>()
        .join(" ");
    println!("\n[!] CORRUPTION DETECTED");
    println!("    Offset: 0x{:08X}", offset);
    println!("    Bytes:  [{}]", hex_found);
    println!("    Error:  {}", msg);
}

