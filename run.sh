#!/usr/bin/env fish

set UV uv
if uname -r | string match -eqi "microsoft"
    set UV uv.exe
end

printf "> JVBERi0xLjUNJeLjz9MNCjM0IDAgb2JqDTw8L0xpbmVhcml6ZWQgMS9MIDI3NjAyOC9PIDM2L0Ug\n" > page-000.txt

# parallel -j 16 $UV run ./monospace.py ../EFTA00400459-{}_2x.png -o page-{}.txt ::: (printf "%03d\n" (seq 1 74))
# command $UV run ./monospace.py ../EFTA00400459-075_2x.png -o page-075.txt --lines 34

# parallel -j 16 $UV run ./monospace.py -d ../EFTA00400459-{}.png -o npage-{}.txt ::: (printf "%03d\n" (seq 1 75))
parallel -j 16 $UV run ./monospace.py -d ../EFTA00400459-{}.png -o page-{}.txt ::: (printf "%03d\n" (seq 1 75))
# Buffer output to allow replacing input without `sponge`
string replace -m1 -r "NCjExNg0KJSVFT0YNCg.*" "NCjExNg0KJSVFT0YNCg==\n" -- (cat page-075.txt) > page-075.txt

cd ./b64-debug
cat ../page-0*.txt | string trim -c "> " | string replace -a \r '' | tee ../recovered.txt | cargo run --release -- --strict > ../recovered.pdf
cd ../

if type -q rewrite
    string join "" < recovered.txt | string replace -ar '(.{76})' '$1\n' | rewrite recovered.txt
end
