---
title: "Intrinsic dimension について"
emoji: "👋"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: []
published: false
---

# はじめに

大規模言語モデル（Large Language Model; LLM）や画像生成系モデル（Diffusion model）などで近年、ユーザー固有の downstream タスクを解かせるための fine-tuning の方法として
Low-Rank Adaptation（LoRA）と呼ばれる手法が人気となっています。この LoRA の手法は低ランク行列を用いて最適化を行うことでパラメータ数を抑えつつ、高精度に fine-tuning できるとしている手法です。先行研究 ^[[Measuring the Intrinsic Dimension of Objective Landscapes](https://arxiv.org/abs/1804.08838), [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255)] で議論されている intrinsic dimension にインスピレーションを受けたと紹介されているため、まず [Measuring the Intrinsic Dimension of Objective Landscapes](https://arxiv.org/abs/1804.08838) をベースに「intrinsic dimension」について調査を行ってみました。


# Intrinsic dimension 

## 導入

[Measuring the Intrinsic Dimension of Objective Landscapes](https://arxiv.org/abs/1804.08838) にて提案されたのが intrinsic dimension であり、本論文では「objective landscape」をよりよく理解するためのツールとして導入されています。
近年の深層学習モデルは膨大な数のパラメータを持っており、高次元空間での最適化に成功したモデルが脚光を浴びている世界となっています。しかしこれまでも何故そのように膨大なパラメータ数を持っていても尚良い精度が出るのか、どんなパラメータ数が効いているのか、本当に必要なパラメータ数はどれくらいなのか、等といった基本的な疑問には回答できていないのが現状だと思います。高次元空間で目的関数がどのような過程を経て最適化されるのか、objective landspace をより理解しようとして intrinsic dimension という考え方が提案されました。

## 留意点

もしかすると intrinsic dimension は業界の中で当たり前の発想だったりするのかもしれないですが、少なくとも論文を読む中で「なぜ intrinsic dimension を導入したのか？これを見て何が嬉しいのか？」といった疑問には答えていなかったように感じました。この指標を用いればより直接的に効果的なモデル構造を作成できるというものでもなく、かといって高精度なモデルが作成できるというものでもないです^[LoRAは本手法を元に作成されたらしいが、どれくらい真面目に先行研究として引いているのかは分からない。]。そのため、なぜこんなことをしているのだろうか？という疑問に関しては最後まで明らかにならないということを、留意点として明記しておきます。

## 考え方

通常のモデルの最適化はそのパラメータ数を $D$ とすると、$\mathbb{R}^D$ における最適化問題を解くことになります。ここでは $D$ 次元空間ではなくランダムに $d$ 次元部分空間（$d<D$） を定義して、その部分空間内での最適化問題を解くこととします。このときに通常の最適化で得られた解と（$\simeq$ 精度）近しいもの、満足の行く解が部分空間内での最適化で得られた場合に、$d$ を intrinsic dimension と呼びます。つまり intrinsic dimension とは、目的関数の最適化においてある許容できる誤差の範囲内で最適化するための最小の部分空間の次元です。定義に関しては [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255) が詳しく、

> An objective function’s intrinsic dimension measures the minimum number of parameters needed to reach satisfactory solutions to the respective objective (Li et al., 2018). Alternatively, the intrinsic dimension represents the lowest dimensional subspace in which one can optimize the original objective function to within a certain level of approximation error.

と記述されています。


## イメージ

部分空間での最適化についてのイメージが Figure.1 に示されているので引用します。ここでは真の最適化問題は $\mathbb{R}^3$ で解く必要があるが、部分空間としてとある $\mathbb{R}^2$ （=2次元平面）を定義したときに、その平面上で最適化（パラメータの更新）を実行するというのがここまでに述べた「部分空間で最適化」するという考え方です。 そしてこのときに満足のいく解が見つかれば $\mathbb{R}^2$ を intrinsic dimension と呼ぶのでした。

![](https://storage.googleapis.com/zenn-user-upload/54a383299e83-20231218.png)


実際に部分空間での最適化を実現するために、以下の表式を導入します：

$$
\theta^{(D)} = \theta_0^{(D)} + P \theta^{(d)}
$$

ここで $\theta^{(D)}$ が最適化したいモデルパラメーター、 $\theta_0^{(D)}$ がモデルパラメーターの初期値、$P$ が $D\times d$ の行列、$\theta^{(d)}$ が部分空間のベクトルです。$\theta_0^{(D)}$ と $P$ はランダムに初期化した後固定、$\theta^{(d)}$ は零ベクトルで初期化し、$\theta^{(d)}$ に関して最適化を行います。ですので、

$$
\frac{\partial f(\theta^{(D)})}{\partial \theta^{(d)}}
$$

の勾配を計算して最適化を行っていくことになります。

$P$ は $D\times d$ の行列であるため、部分空間 $\mathbb{R}^d$ から $\mathbb{R}^D$ への射影子として考えることができます。つまり、ランダムに部分空間を定義する（$P$ はランダムに初期化するため）ということです^[この辺りはあまり厳密に議論されているような気はしないですが、単純に「$d<D$ が成り立つような $\mathbb{R}^d$ を $\mathbb{R}^D$ の部分空間だ」としている以上の意味は持っていないと感じました。]。ランダムに部分空間を定義して最適化がうまく行けばそれを intrinsic dimension と呼ぶのです。

# 実験

ここまでで定義した intrinsic dimension の考え方を用いた各種実験の結果も見ていきたいと思います。ただし $d_{100}$ は全パラメータ空間で最適化した場合の精度と等しくなるような intrinsic dimension、$d_{90}$ はその 90% の精度が達成された場合の intrinsic dimensino を指すこととします。上述の説明で「満足のいく解が得られた場合に」とお茶を濁してきたことに対して、一定の定義付けをしておきました（論文で使用されている表現方法です）。早速結果ですが、全結合層と LeNet を用いて MNIST の10クラス分類を解くモデルに対して以下のような結果になっています：

![](https://storage.googleapis.com/zenn-user-upload/3e3d55235292-20231218.png)

横軸が intrinsic dimension で縦軸が validation acc. です。この図が示しているのは元々のパラメーター数が $10^4$ 〜 $10^5$ オーダーであったのに対して、$10^2$ オーダーのパラメータ数で 90% の精度を達成しているということです。もちろん全パラメータ数を使って最適化した場合の方が精度は良いのですが、少ないパラメータ（小さい部分空間）で最適化した場合でも遜色ない精度が達成できているということが驚くべき結果かと思います。


# まとめ

ランダムに定義した部分空間^[$P$をランダムに初期化しているという意味であり、特に深い意味はなさそうでした。]で最適化を行った場合でも、通常の最適化と遜色ない精度を達成できるということが示されました。intrinsic dimension を指標として使用することでモデルへの理解がより一層深まるとしています。論文の締め（§4. Conclusions and Future directions） では

> Further work could also identify better ways of creating subspaces for reparameterization: here we chose random linear subspaces, but one might carefully construct other linear or non-linear subspaces to be even more likely to contain solutions.
> future work として部分空間の定義方法についてよりより手法が見つかるだろう。今回はランダムな線形部分空間を選んだが、他の線形または非線形の部分空間を注意深く構築することで、さらに解を含む可能性が高くなるかもしれない。

と議論しており、そもそも部分空間の定義方法については課題があると言っています。これから、intrinsic dimension が何なのか、よりよい部分空間の定義方法などの研究が進んでいくと面白い結果が出てきそうで楽しみです。



# 参考文献

- [Measuring the Intrinsic Dimension of Objective Landscapes](https://arxiv.org/abs/1804.08838)
- [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)