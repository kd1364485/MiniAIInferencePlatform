# SystemProbe

**何を作ったか**: Linux の `/proc` を読んで、メモリ・CPU・ネットワーク・ディスク・スレッド数を取り出す C++ の小さなツール(`src/`)。
**どう確かめたか**: 文字列の解釈(`parse.h`)と実際の読み取り(`collect.h`)を分け、それぞれにテスト(`test/`)を書いた。
**何が分かったか**: 各読み取りの時間を `bench/` で測った(`results.txt`)。たとえば、`/proc` を実際に開いて読む時間は、メモリ上の文字列を解釈する時間より桁違いに大きい。
`strace` / `perf` / `gdb` での観察は、末尾の「テストと性能測定を追加」にある。
**この下は、日々の学習ログ**(M0〜M6、古い順)。最新の整理は末尾にある。

## 2026/09/25　M0
### やったこと
WSL環境構築　CMakeプロジェクト作成、/proc/version読んで表示

### 学んだこと
/procとはカーネルが内部情報をファイルとして公開する疑似ファイルシステム
これを使うことで特別なAPIとかは必要なく普通のファイル読み込みで内容を読み取れる


forkとは自分自身のコピーのプロセスを作るシステムコール。
dupとは既存のfdを複製するシステムコール

エントリとはカーネル内の fdテーブルの1行分の記録

File descriptor（fd）
fdテーブルのエントリ = 「開いたファイルへの参照」を保持するスロット。fd番号はその添字。read/writeはfd番号経由で対象を特定する




RAII 「リソースの取得をオブジェクトの初期化に紐づける」

ルールは単純：

コンストラクタでリソースを取得（openする、メモリを確保する、ロックを取る）
デストラクタで解放（closeする、解放する、アンロックする）
デストラクタはスコープを抜けるとき自動で呼ばれる。例外が飛んでも必ず呼ばれる

### つまずき

CMakeLests.txtの置き場所（ルートに置く理由＝src/main.cppへの相対パス）

===============================================================

## 2026/09/25　M1
### やったこと

/proc/meminfo をパースして Total/Used/Available/Free を MiB で表示
free -m と比較して検証した

### 学んだこと

/proc/meminfo のフォーマット：Key: value kB という行の集まり

MemAvailable vs MemFree の違い 

MemFree：完全に空っぽのメモリ
文字通り「何も入っていない」物理メモリです。

Linuxの設計思想：空きメモリはもったいない
Linuxは使っていないメモリをキャッシュとして勝手に活用する


MemAvailable：実質使える量
カーネルが「新しいプログラムを始めるなら、スワップなしでこれだけ使える」と見積もった値です。
MemAvailable ≈ MemFree + すぐ解放できるキャッシュ等



パースのパターン：getline で1行 → meminfo は「1行 = 1レコード」のフォーマット。getline で行を区切ってから istringstream で分解することで、1行の中身だけでパースが完結する。もしファイル全体に file >> k >> value >> unit すると、HugePages_Total: 0 のような単位なしの行でずれて、次の行のキーを unit として読んでしまう

kB → MiB の換算：value / 1024


MiB は Mebibyte（メビバイト）。単位の話

### つまずき

<sstream> のinclude忘れ → incomplete type エラー
教訓：「incomplete type = ヘッダ不足を疑え」「機能ごとにヘッダが分かれている」

===============================================================
## 2026/10/02　M2

### やったこと
/proc/cpuinfo から Model name（CPU名）と Logical cores（論理コア数）を取得

/proc/statからCPU使用率を計算して表示

nprocとtopで値を突き合わせて検証した

### 学んだこと
/proc/cpuinfoのフォーマット：key : value (コロン区切り)
/proc/meminfoのKey: value kBとは別形式
starts_with で行頭判定 + find(':') と substr で右側を取り出すパターン

/proc/stat の値は何か：jiffies単位の起動からの累積カウンタ。瞬間値ではない
→ だから「1回読むだけでは使用率が出ない」。500ms 挟んで2回読み、差分を取る
→ 式： usage = (total_d - idle_d) / total_d * 100
idle の定義：idle + iowait。iowait は「I/O待ちでCPUが遊んでいる時間」なので「仕事していない時間」に含める
struct でまとめて返す：8個のフィールドを CpuTimes にまとめ、関数は値で返し、計算側は const& で受ける設計
std::this_thread::sleep_for + <chrono>：計測区間を作るためのsleep。std::chrono::milliseconds(500) の書き方

### つまずき
total_time() を空のまま書きかけで放置していた → -Wreturn-type 警告。非void関数で return を忘れると未定義動作になる(進捗メモもっと具体的にどこまでかもうちょっと詳しく書くべきだった)

写経ミス："%\n" でModel名末尾に % 混入、" cores" のスペース抜けで 20cores → 出力を目視確認する癖の重要性

Clock skew 警告は Windows側から編集したときの時刻ズレ（実害なし）

===============================================================
## 2026/10/02　M3

### やったこと
std::filesystem::directory_iteratorで/proc直下を列挙し
ディレクトリ名が全部数字のもの＝プロセスとしてカウント

各/proc/<pid>/statusのState行から状態コードを取得し、
std::map<char,int>で状態別に集計

ps -e --no-headers | wc -l とps -eo statで検証

### 学んだこと

/proc/<pid>/：プロセスごとの情報がディレクトリとして見える。中に status stat cmdline fd/ などがある（ls /proc/1/ で実物を見た）

C++の文は関数の中にしか書けない：ファイルのトップレベルは宣言専用（つまずき欄と対になる学び）

directory_iterator の使い方：entry.is_directory() / entry.path().filename().string() で名前を取る

std::map で集計：states[key]++ で勝手に0初期化される。ループは structured binding for (const auto& [k, v] : states) で読む

状態コードの意味：R=実行中、S=割り込み可能なsleep（ほとんどのプロセスはこれ）、Z=zombie（終了したのに親が後始末してない）

計測値はその瞬間のスナップショット：psと1〜2ずれたのはバグではなく、プロセスが常に生滅しているから。しかもps実行中はps自身やパイプのコマンドもRに数えられる → 「観測者が観測対象を汚す」問題

### つまずき
集計コードだけ書いて出力忘れた　忘れないように気を付けよう

line[7]とやっていたがこの固定位置取りはState:\tSのレイアウント前提
なのでfind_first_not_ofで空白を飛ばす方が堅い

===============================================================
## 2026/10/02　M4

### やったこと
スレッド数を２方式で取得し比較/proc/<pid>/task/ 列挙 vs /proc/<pid>/status の Threads: 行

vector<pair> + sort（ラムダで降順）でTop5表示

プロセス名は /proc/<pid>/comm から取得

### 学んだこと
Linuxではスレッドも[task]→　/proc/<pid>/task/<tid>に見える
だから２方式が一致する

directory_iterator は例外を投げる（走査中に対象プロセスが消えると filesystem_error）→ try/catch 必須

ifstream は open 失敗で例外を投げない（デフォルトで例外無効）→ if (!file) で十分

std::pair / push_back({a,b}) / sort にラムダで比較ルールを渡す

comm はカーネルが15文字に切り詰める（psも同じ）。stoi で文字列→数値変換


### つまずき

States: ラベルと値の間にTop5を挟んで出力順が壊れた → 出力はコード順
ifstream に不要な try/catch を書いた → 何が例外を投げるかは関数次第

unattended-upgr みたいに名前が切れるのはなぜ？
アンサー：表示名は /proc/<pid>/comm から取っている
comm はカーネル側で最大15文字に切り詰められる（TASK_COMM_LEN = 16、末尾1バイトは終端用）
つまりパース処理の問題ではなく、取得元のデータ自体が最初から切れている
証拠に ps でも同じく unattended-upgr と表示される
===============================================================
## 2026/10/02　M5

### やったこと
/proc/net/devをパースしてNICごとのRX/TX　累積バイト数を表示

CPU使用率と同じ500ｍｓ窓で差分を取りKiB/sを算出

生データ(cat /proc/net/dev)と突き合わせて検証

### 学んだこと
差分計測は汎用パターン：CPU・ネットワークで同じ型が使える（structでカウンタ保持 → 2回読み → 差分 ÷ 時間）
find / substr / find_first_not_of での文字列処理、istringstream で16個連続読み
名前で対応させる理由：インターフェースの出現順は保証されない（添字で対応させるとズレる）



### つまずき
タイポ3連発（NetDe、dev1、dev2）→ コンパイラの did you mean 提案が便利
-> 演算子：b->x = (*b).x。.は本体/参照、->はポインタ経由。find_dev がポインタを返すのは nullptr（無）を表せるから
/proc/net/dev の形式：ヘッダ2行 + 名前: 数値16個（RX 8個 + TX 8個）

===============================================================
## 2026/10/02　M6

### やったこと
/proc/diskstats パース → デバイスごとの reads・R/W累積KiB表示。実値と突き合わせ


### 学んだこと

/proc/diskstats の形式：major minor name のあと数値列。4番目=reads completed、6番目=sectors read、8=writes、10=sectors written
1セクタ = 512バイト。KiB換算は sectors * 512 / 1024（= /2）
パースは : 分割不要で >> 連鎖だけ（net/devより単純）
ダミー変数で「要らないフィールドを読み飛ばす」技術（M5と同じ）


### つまずき
istringstream を getline の外で作っていた → issは1行ごとに作る。戻り値型の忘れなど



1. プロセスとは
実行中のプログラムの実体でプログラム（ディスク上のバイナリ）を起動すると、カーネルが：

メモリ空間（後述のvirtual memory）
fdテーブル
task_struct（管理情報の台帳）
を一式用意し「プロセス」として登録。/proc/<pid>/ はその台帳をファイルとして覗かせている窓。

2. Virtual memoryとは
各プロセスが「自分専用のメモリ空間」を持っているように見せる仕組み。

プロセスが使うアドレスは仮想アドレス。実際のRAMの番地ではなく
CPUのMMUとカーネルのページテーブルが「仮想→物理」の変換をアクセスのたびにやる
プロセスAとBが同じ仮想アドレス 0x400000 を使っても、指す物理メモリは別

3. fd 
「スロット」を fd と言うか、fd番号を言うかのニュアンスだけ。正確には：

fd（ファイルディスクリプタ）= fdテーブルの添字（番号）
fdテーブルのエントリ（スロット）= 開いたファイルへの参照を保持する行
read(fd, ...) とするとカーネルは「そのプロセスのfdテーブルのfd番目」を見て対象を特定する
そして重要な応用：fdが指すのはファイルだけじゃない。ソケット・パイプ・デバイスもfdとして扱われる（"everything is a file"）。Phase 2でsocketを触るときに直結します。

4. RAII

コンストラクタで取得、デストラクタで解放
スコープを抜ければ自動でデストラクタが呼ばれる
例外が飛んでも必ず呼ばれる

解放忘れ・例外時のリーク問題から解放される

===============================================================

## 2026/11/02　テストと性能測定を追加

(この節の実装と測定は Claude Code の支援で追加した。上の M0〜M6 は元の学習ログ。)

ロードマップの Phase 1 には「unit test、gdb、strace、perf + benchmark」まで持っていく、とある。
unit test と benchmark を追加し、strace / perf / gdb で測定結果の理由を調べ、見つかった点を 1 つ改善した。

### 構成の変更

M6 までは `src/main.cpp` に、/proc を開く処理・文字列を読み解く処理・表示がすべて入っていた。
文字列を読み解く部分を、ファイルから切り離した。

```
src/parse.h        /proc のテキストを読み解く(std::istream を受け取るだけ。ファイルを開かない)
src/collect.h      /proc を実際に開いて、parse.h で読み解く
src/main.cpp       集めた値を表示するだけ(出力の形は元のプログラムと同じことを確認済み)
test/              単体テスト(parse_test: 疑似 /proc で、collect_test: 実機の /proc で)
bench/bench.cpp    所要時間のベンチマーク
bench/syscalls.cpp strace / perf で観察するための、1 つの処理だけを繰り返すプログラム
```

理由: `parse.h` が istream を受け取る作りなら、/proc の中身を模した文字列で「毎回同じ結果になるテスト」が書ける。
実機の /proc を使うテストは値が環境で変わるので、あり得る範囲や、別の方法との一致で確かめる。

### ビルドとテスト

```bash
cmake -S . -B build          # 指定がなければ Release でビルドされる
cmake --build build -j
cd build && ctest            # parse_test と collect_test
./systemprobe                # 本体
./bench 2000                 # ベンチマーク(反復回数)
```

外部ライブラリは使わず、最小の CHECK マクロ(`test/check.h`)で書いた。

### テスト

- `parse_test`(58 項目): meminfo / cpuinfo / stat / net/dev / diskstats / status の各パーサーを、正常な入力と壊れた入力(空、キーなし、数値が足りない、単位なしの行など)で確かめる。CPU 使用率は「iowait は使用ではなく idle 側」「変化がなければ 0(0 除算しない)」も確かめる。
- `collect_test`: 実機で、メモリ・CPU の値が妥当な範囲か、**スレッドを 4 本足すと、`task/` を数える方式も `Threads:` 行を読む方式も、ちょうど 4 増える**こと、存在しない PID で落ちないこと、を確かめる。

**テストが本当に間違いを検出できるか** も確かめた。コードをわざと壊すと、テストは失敗した:

| わざと入れたバグ | 結果 |
|---|---|
| `idle_time` から iowait を外す | 2 項目が失敗(56/58) |
| `is_pid_name` の空文字チェックを外す | 1 項目が失敗(57/58) |
| diskstats の書き込みセクタ数の列を 1 つずらす | 1 項目が失敗(57/58) |

テストを書く過程で見つかった小さな不具合: `is_pid_name("")`(空文字)が `true` を返していた(`std::all_of` は空の範囲で `true` を返すため)。
実際の /proc には空の名前がないので、動作には影響しなかったが、明示的に除いた。

### ベンチマーク

環境: Intel Core i7-12700H(論理 20 CPU)、WSL2(Linux 6.6.87.2)、g++ 13.3、Release(-O2)、測定時のプロセス数は 46。
`./bench 2000` を 3 回。以下は各回の **p50 の範囲**(µs)。PC の負荷でぶれるので、1 回の値ではなく範囲で示す。生の出力は `results.txt`。

**1. 1 回の収集にかかる時間(ファイルを開いて読む + 解釈)と、解釈だけの時間**

| 処理 | ファイルから(p50) | 解釈だけ・メモリ上(p50) |
|---|---|---|
| meminfo | 4.0 〜 4.6 | 0.6 |
| cpu_model_name | 16.5 〜 17.9 | 0.6 〜 0.7 |
| logical_cpus(cpuinfo を数える) | **154 〜 157** | 7.1 〜 7.7 |
| /proc/stat(CPU 時間) | 8.2 〜 8.7 | 0.5 |
| net/dev | 12.7 〜 13.6 | 4.7 〜 5.1 |
| diskstats | 37.6 〜 43.0 | 9.1 〜 10.1 |

多くの処理で、時間の大半は「ファイルを開いて読む」側(システムコールとカーネルの処理)で、文字列の解釈は小さい。
例外は net/dev と diskstats で、解釈も全体の 2.5〜4 割を占める(`istringstream` を行ごとに作るため。未最適化)。

**2. 全プロセスのスレッド数を数える 2 方式(1 周の p50、µs)**

| 方式 | C++(ifstream / directory_iterator) | 素のシステムコールだけ |
|---|---|---|
| `/proc/<pid>/task/` の中身を数える | 137 〜 150 | **76 〜 83** |
| `/proc/<pid>/status` の `Threads:` 行を読む | 225 〜 246 | **181 〜 198** |
| 比(status ÷ task) | 1.50 〜 1.73 | 2.37 〜 2.40 |

3 回とも **`task/` を数える方式のほうが速かった**。事前の予想は逆だった(ディレクトリ一覧のほうが遅いだろう、と)。測ってみて外れたので、`collect.h` のコメントも直した。

**3. 全体(メモリ・CPU・全プロセス・ネットワーク・ディスクを 1 周。500ms の待ちを除く、p50)**: 約 1.02 〜 1.13 ms。

### strace・perf・gdb で、理由を調べた

生の出力は `results_tools.txt`。

**(a) スレッド数の 2 方式: 差はカーネル側にあり、システムコールの回数では説明できない**

`strace -c` でシステムコールを数えた(47 プロセス、50 周)。

| 方式 | 1 プロセスあたりのシステムコール |
|---|---|
| status | **約 3 回**(openat、read、close) |
| task | **約 7 回**(openat、fstat、fcntl ×2、getdents64 ×2、close) |

回数は task のほうが 2 倍以上多いのに、速いのは task。では差はどこから来るのか。
C++ のストリームの処理(ユーザー空間)が原因かもしれないので、`ifstream` と `directory_iterator` を外し、`open` + `read`(または `getdents64`)+ `close` だけで測り直した(上の表の右列)。
**ストリームを外しても status は task の約 2.4 倍遅い**ので、差の主因はカーネル側(`/proc/<pid>/status` を読むたびにカーネルが文章を組み立てるコスト)だと分かった。
C++ のストリームは、どちらの方式にも 40〜70µs ほど上乗せしている(1 周あたり)。
カーネルの中の何がどれだけ重いのかまでは、perf でカーネル側が見えないため確かめていない(下の「限界」)。

**(b) logical_cpus が重い理由: cpuinfo は何度も read する**

`strace -T` で 1 回の読み取りを見ると、`/proc/cpuinfo` は **read が 11 回**(うち最後の 1 回は EOF)で、1 回の read で返るのは約 2.7KB = **約 2 CPU 分**だった。
20 CPU なら 10 回の read になるので、**論理 CPU が増えるほど遅くなる**。`getrusage` でも、時間の約 91〜95% はカーネル側(sys)だった。
一方、`sysconf(_SC_NPROCESSORS_ONLN)` は `/sys/devices/system/cpu/online` という 1 行のファイルを開くだけで、システムコールは 3 回(open、read、close)。
`cpu_model_name` が約 17µs で済むのは、最初の `model name` で読むのをやめて、2 回目以降の read をしないから。

**(c) perf stat(ソフトウェアイベント)**

どの処理でも `context-switches` は 0 で、`page-faults` は反復回数(100 回でも 2000 回でも)にかかわらず約 125〜133 で一定だった。
つまり、繰り返しごとのメモリ確保や、スレッドの切り替えは、遅さの原因ではない。

**(d) gdb: パーサーの流れを追った**

`-DCMAKE_BUILD_TYPE=Debug` でビルドし、`parse_netdev_line` にブレークポイントを置いて確かめた。
呼び出しは `main` → `read_netdev` → `parse_netdev` → `parse_netdev_line` の順。1 行目 `    lo:   27943 ...` では、`colon = 6`、`substr(0, colon)` の結果は `"    lo"`(**先頭に空白が付く**)。
だから、名前を取り出す前に `find_first_not_of(" \t")` で空白を取り除く処理が要る。戻り値は `true`。

### 見つけた問題と、1 つ改善

**問題**: `logical_cpus()`(論理 CPU 数)が約 155µs かかり、1 回の収集の中で突出して重い。解釈そのものは約 7µs しかない。(b)のとおり、原因は cpuinfo を何度も read するカーネル側のコスト。

**改善**: `sysconf(_SC_NPROCESSORS_ONLN)` を使う `logical_cpus_online()` を追加した。値は同じ(テストで一致を確認)。

| 方法 | p50(µs) |
|---|---|
| `logical_cpus()`(cpuinfo を開いて数える) | 154 〜 157 |
| `logical_cpus_online()`(sysconf) | 2.0 〜 2.2 |

**約 70 〜 78 倍**速い。ただし、`systemprobe` 本体は 500ms 待つ使い捨てのコマンドなので、この差は体感できない。
一定間隔で繰り返し動かす監視ツールにしたときに効く最適化なので、`main.cpp` はこれまでの `logical_cpus()`(/proc を読む版)のままにしてある。

### 測定の限界

- **perf はソフトウェアイベントの `perf stat` しか使えなかった。** WSL2 のカーネルに合う perf がなく、Ubuntu 付属の別の版(6.8)を直接実行した。ハードウェアカウンタ(cycles、instructions)は `<not supported>`。また `perf_event_paranoid=2` なので、カーネル側のサンプルが取れず、`perf record` で「カーネルの中のどこが重いか」は見られなかった。この設定は変更していない。
- **strace は 1 回ごとに約 10µs の監視コストが乗る**。そのため、時間の内訳ではなく、**回数**だけを根拠にした。
- 負荷の影響を受ける(3 回の p50 に、最大 1.2 倍ほどの差が出た項目がある。値が 1µs 前後の項目では、もっと大きい)。WSL2 上の測定で、実機の Linux とは異なる可能性がある。
- `getrusage` の user / sys の内訳は、短い処理を少ない反復で測ると実行ごとに大きくぶれた(全プロセス 1 周では sys が 42〜100% になった)。そのため、割合が安定した `logical_cpus` の項目だけを `bench` に残した。
- WSL と Windows の時計が約 3 秒ずれていて、ビルド時に `Clock skew detected` の警告が出る。結果には影響しない。

### 再現手順

```bash
# strace: システムコールの回数(1 周あたり = total ÷ N)
strace -c -f -S calls ./syscalls task 50
strace -c -f -S calls ./syscalls status 50
strace -T -e trace=openat,read,close ./syscalls cpuinfo 1

# perf(WSL2 では、版の合うものを直接指定する)
PERF=$(ls /usr/lib/linux-tools/*/perf | head -1)
$PERF stat -e task-clock,context-switches,page-faults ./syscalls cpuinfo 2000

# gdb(Debug ビルド)
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug && cmake --build build-debug
gdb -batch -ex 'break probe::parse_netdev_line' -ex run -ex bt -ex 'print line' build-debug/systemprobe
```