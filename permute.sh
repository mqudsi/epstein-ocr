#!/bin/sh
#![allow(missing_abi)] /*
# rust self-compiler by M. Al-Qudsi, licensed as public domain or MIT.
# See <https://neosmart.net/blog/self-compiling-rust-code/> for info & updates.
OUT=/tmp/$(printf "%s" $(realpath $(which "$0")) | md5sum | cut -d' '  -f1)
MD5=$(md5sum "$0" | cut -d' '  -f1)
(test -x "${OUT}" -a "${MD5}" = "$(cat "${OUT}.md5" 2>/dev/null)" ||
(grep -Eq '^\s*(\[([^][]*)])*\s*fn\s+main\b' "$0" && (rm -f ${OUT};
rustc "$0" -o ${OUT} && printf "%s" ${MD5} > ${OUT}.md5) || (rm -f ${OUT};
printf "fn main() {//%s\n}" "$(cat $0)" | rustc - -o ${OUT} &&
printf "%s" ${MD5} > ${OUT}.md5))) && exec ${OUT} "$@" || exit $? #*/

use std::fs;
use std::process::Command;

#[derive(Copy, Clone, Debug, PartialOrd, Ord, PartialEq, Eq)]
pub enum Input {
    TrainTop,
    TrainBot,
}

fn main() {
    let top_template = "train_top.in";
    let bot_template = "train_bot.in";
    let top_path = "train_top.txt";
    let bot_path = "train_bot.txt";
    let runner = "./try.sh";

    // Read original top/bottom training files
    let top_content = fs::read_to_string(top_template)
        .unwrap_or_else(|_| panic!("Failed to read {}", top_path));
    let bot_content = fs::read_to_string(bot_template)
        .unwrap_or_else(|_| panic!("Failed to read {}", bot_path));

    let mut top_chars: Vec<char> = top_content.chars().collect();
    let mut bot_chars: Vec<char> = bot_content.chars().collect();

    // Identify all 1/l positions
    let mut slots = Vec::new();

    // Skip the first 4 'l's, which come from English text and
    // are sure to be correct.
    for (i, &c) in top_chars.iter().enumerate().skip(4) {
        if c == '#' {
            slots.push((Input::TrainTop, i));
        }
    }
    for (i, &c) in bot_chars.iter().enumerate() {
        if c == '#' {
            slots.push((Input::TrainBot, i));
        }
    }

    let n = slots.len();
    let combinations = 1u128 << n;

    println!("Found {} slots across both files.", n);
    println!("Total permutations to generate: {}", combinations);

    // Try all possible permutations (not combinations, obviously!)
    for i in 0..combinations {
        for bit_idx in 0..n {
            let bit = (i >> bit_idx) & 1;
            let new_char = if bit == 0 { '1' } else { 'l' };

            let (file_id, char_idx) = slots[bit_idx];
            if file_id == Input::TrainTop {
                top_chars[char_idx] = new_char;
            } else {
                bot_chars[char_idx] = new_char;
            }
        }

        // Write the modified content back to the files
        let new_top: String = top_chars.iter().collect();
        let new_bot: String = bot_chars.iter().collect();

        fs::write(top_path, new_top).expect("Failed to write to train_top.txt");
        fs::write(bot_path, new_bot).expect("Failed to write to train_bot.txt");

        // Call ./try.sh runner with the generation/sequence number
        let seq = format!("{:03}", i);
        println!("Running permutation {i}/{combinations}...");

        let status = Command::new(runner)
            .arg(&seq)
            .status()
            .unwrap_or_else(|_| panic!("Failed to execute {}", runner));

        if status.success() {
            println!("{seq}: {runner} exited with status code zero!");
        } else {
            eprintln!("{seq} {runner} returned non-zero exit code.");
        }
    }

    println!("All permutations completed.");
}

// vim: set ft=rust :
