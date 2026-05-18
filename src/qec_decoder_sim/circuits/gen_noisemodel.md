問題なし。実装完了です。

実装の概要をまとめます。

変更内容
gen/_core/_noise.py

NoiseModel.__init__ に idle_x_error: float = 0 パラメータを追加
_append_idle_error に X_ERROR のidle noise処理を追加
NoiseModel.bitflip(p) static methodを追加：
idle_x_error=p（ゲートレベル回路で使用する場合のidle X_ERROR）
全1q/2qゲートに空の NoiseRule()（noisy_circuit使用時にValueErrorを回避）
測定・リセットにも空の NoiseRule()
circuits/_topological_memory_circuits.py

NoiseModelName に "bitflip" を追加
_NOISE_MODEL_BUILDERS に "bitflip": gen.NoiseModel.bitflip を追加
_make_bitflip_code_capacity_circuit() を新設：
NoiseRule(after={"X_ERROR": p}, flip_result=0) を使用（X_ERRORノイズ・測定は完全）
make_phenom_circuit() を呼び出す → MPP測定 + ラウンド間X_ERROR = 真のcode capacityモデル
surface_code_Z：make_surface_code_layout → make_phenom_circuit
superdense_color_code_Z：make_color_code_layout_for_superdense → make_phenom_circuit
設計の要点
uniform/si1000 がゲートレベル回路（CNOT等を含む）にノイズを追加するのに対し、bitflip はphenomenological回路（MPP測定）を生成します。これにより：

各syndrome extraction roundの後に データ量子ビットにX_ERRORを挿入
測定は完全（flip_result=0）
これが真のcode capacity noise modelの定義と一致