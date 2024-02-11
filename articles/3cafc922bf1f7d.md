---
title: "VAE 理解までの道のり"
emoji: "🐥"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: []
published: false
---


# はじめに

Variational Autoencoder（VAE）と呼ばれるモデルが提案されたのは 2014 年の [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf) です。ただ VAE の何がすごいのか、Stable Diffusion などの前処理でもいまだ VAE が使用されていますが、何が凄いのか、よく分からなかったので本稿でその背景に迫ってみたいと思います。


Autoencoder との比較記事などがあり、「え、VAEは AutoencoderのSOTAなの？」と思ったりしていたのですが、特に元論文にそのような記載もないので混乱していました。



なぜ潜在変数を考えているのか？



Autoencoder に非線形変換を加えたモデルが VAE です。そのため複雑な構造をもつデータのモデル化が可能になります。


# 流れ

- 最尤推定とベイズ推定
  - 点推定か確率推定か？ $\theta$ か、$p(\theta)$ か？
  - 最尤推定とは likelihood を "最大化" することでパラメータを求める（likelihoodを使うことがイコール最尤法ではない）


- 変分推論
  - ELBO
- 平均場近似
- EMアルゴリズム
- 変分EMアルゴリズム
  - 何が変分なのか？

- 最尤推定的な取り扱い
- MAP


# Variational Autoencoder


$\beta$ VAE、FactorVAE、$\beta$-TCVAE
DIPVAE
JointVAE
CascadeVAE

#


最尤法ではパラメータ $\theta$ は決定論的に値が決まるものとして扱います
ベイズ的取り扱いでは、パラメータ $\theta$ は確率変数として扱います

- fully Beysian model では全ての未知変数には事前分布が与えられる


ベイズ推論を用いた手法では確率モデルを設計することになります。

$$
p(X, Z)
$$

学習や予測などの具体的な課題は事後分布 $p(Z|X)$ を計算することで実現できます。


## 線形モデル

線形モデルを例に、最尤推定的な取り扱いと、ベイズ的な取り扱いについて考察しておきたいと思います^[「ベイズ的な取り扱い」のしっくりくる単語が無いような気がしますが...。]。

### 最尤推定

パラメータ $\theta$ が求まる

### ベイズ推定


事後分布 $p(\theta|X)$ が求まる



#  混合ガウス分布

潜在変数を持つモデルについて、混合ガウス分布を題材にしてみます。


確率分布を線形結合して作る確率分布を混合分布（mixture distribution）と呼びます。このときにガウス分布を使用したモデルを混合ガウス分布と呼び、$K$ 個のガウス分布の重ね合わせ

$$
p(\rm{x}) = \sum_{k=1}^K \pi_k N(\rm{x}|\mu_k, \Sigma_k)
$$

で表されます。それぞれ個別に平均と共分散のパラメータを持っています。

十分な数のガウス分布を用いて、係数と平均・共分散を調節すれば、ほぼ任意の連続な密度関数を任意の精度で近似できます。


## 問題

混合ガウス分布が有用であるということは分かったのですが、ではそれらのパラメータをどのように求めれば良いのでしょうか？ここでEMアルゴリズムと呼ばれる枠組みを用いることができます。


対数尤度関数は

$$
\ln p(\rm{X}|\pi, \mu, \Sigma) = \sum_{n=1}^N \ln \left\{\sum_{k=1}^K \pi_k N(\rm{x}_n|\mu_k, \Sigma_k) \right\}
$$

となるのですが、尤度関数はもはや解析的に計算できなくなってしまっています。そこで混合ガウス分布に潜在変数を導入して、EMアルゴリズムの枠組みを使用することでパラメータ推定が可能となります。







# 


- $\ln p(X|\theta)$ と $\ln p(X)$ について
  - 最尤法では前者、ベイズ推定では後者を扱う
  - $\ln p(X|\theta)$ を書き下したときに、$\theta$ と $q$ とで最適化する必要があるので、EMアルゴリズムが強力に働く






#  EM アルゴリズム

EMアルゴリズムを題材として、最尤推定とベイズ推定の違いについて更に深ぼって見ていきたいと思います。根本の違いはそれぞれの推定方法の差異であるモデルパラメータの取り扱いに起因します。



## 最尤法的な取り扱い （通常のEMアルゴリズム）


EM アルゴリズムは最尤推定の計算方法の一つで、潜在変数を持つモデルの likelihood を最大化するパラメータ $\theta$ を求めることを目的としています^[混合ガウス分布はEMアルゴリズムを使用するために、確率的に解釈して $p(z)$ を導入していました。もともと潜在変数を持っていなくても、EMアルゴリズムの枠組みに合わせてモデルを修正することもあるということです。]。

観測した変数と潜在変数のどちらの情報も分かっていれば（$X, Z$ は complete dataset と呼ばれます）^[こうなると最早 $Z$ が "潜在" 変数ではないのですが...。]、

$$
\argmax_\theta \ln p(X, Z|\theta)
$$

の joint distribution $p(X, Z|\theta)$ の対数尤度からパラメータ $\theta$ を決定するのが straightforward な手法です。ただし実際には $Z$ がどのような値を取って $X$ と関係しているか分からないです（$X$ だけだと incomlete data と呼ばれます）。

$$
\ln p(X, Z|\theta) = \ln \left( \Sum_Z p(X, Z|\theta) \right) = \ln p(X|\theta)
$$

ここで $p(X|\theta)$ の最適化は難しいが^[混合ガウス分布における $\ln p(\rm{X}|\pi, \mu, \Sigma) = \sum_{n=1}^N \ln \left\{\sum_{k=1}^K \pi_k N(\rm{x}_n|\mu_k, \Sigma_k) \right\}$]、$p(X,Z|\theta)$ の最適化は比較的簡単である場合を想定します。潜在変数の分布 $q(\rm{Z})$ を導入する^[「ベイズ推定 = 事前分布を導入する」と思っていたりすると「EMアルゴリズムは最尤推定の計算方法なのに、事前分布である $q(\rm{Z})$ を導入する...？」と混乱してしまいますが、最尤推定とはパラメータ $\theta$ の決定的な値を求める手法であるということを改めて確認しておきましょう。]ことで、対数尤度を次のように変形することができます：


$$
\begin{aligned}
\ln p(\rm{X}|\theta) 
&= \ln \frac{p(\rm{X}, \rm{Z}|\theta)}{p(\rm{Z}|\rm{X}, \theta)} \\
&= \ln \left\{ \frac{p(\rm{X}, \rm{Z}|\theta)}{p(\rm{Z}|\rm{X}, \theta)}\frac{q(Z)}{q(Z)} \right\}\\
&= \ln \left\{ \frac{p(\rm{X}, \rm{Z}|\theta)}{q(Z)}\frac{q(Z)}{p(\rm{Z}|\rm{X}, \theta)} \right\}\\
&= \ln \frac{p(\rm{X}, \rm{Z}|\theta)}{q(Z)} - \ln \frac{p(\rm{Z}|\rm{X}, \theta)}{q(Z)} \\
\sum_{{\rm{Z}}}q({\rm{Z}}) \ln p(X|\theta) 
&= \sum_{{\rm{Z}}} q({\rm{Z}}) \ln \frac{p({\rm{X}}, \rm{Z}|\theta)}{q(Z)} - \sum_{\rm{Z} q(\rm{Z})} \ln \frac{p(\rm{Z}|\rm{X}, \theta)}{q(Z)} \\
\end{aligned}
$$

よって

$$
\ln p(X|\theta) = 
\sum_{\rm{Z}} q({\rm{Z}}) \ln \frac{p(\rm{X}, \rm{Z}|\theta)}{q(Z)} - \sum_{\rm{Z}} q({\rm{Z}}) \ln \frac{p(\rm{Z}|\rm{X}, \theta)}{q(Z)} 
$$

と変形することができます。
1行目では確率の乗法定理を使用し、2行目では $q(Z)$ を導入しました^[$p(Z)$ とはまた別の形の分布です。]。また5行目では両辺に $q(\rm{Z})$ を掛けて、$\rm{Z}$ に対して和を取ると $\sum_{\rm{Z}} q(\rm{Z})=1$ であることを用いています。

この式は以下のように表現することができます。

$$
\ln p(X|\theta) = L(q(Z), \theta) + \rm{KL} (q||p)
$$

第二項目は KL ダイバージェンスであり $\geq 0$ の量であることを踏まえると、

$$
\ln p(X|\theta) = L(q(Z), \theta) + {\rm{KL}} (q||p) \geq L(q(Z), \theta)
$$

対数尤度の下界が $L(q(Z), \theta)$ で表現できます。

### イェンセンの不等式を使うパターン

イェンセンの不等式を用いることで、対数尤度の下界をより直接的に求めることができます。

$$
\begin{aligned}
\ln p(X|\theta)
&= \ln \sum_Z p(X,Z|\theta) \\
&= \ln \sum_Z q(Z)\frac{p(X,Z|\theta)}{q(Z)} \\
&\geq \sum_Z q(Z) \ln \frac{p(X,Z|\theta)}{q(Z)} \equiv L(q(Z), \theta)\\
\end{aligned}
$$

以上より対数尤度の下界が求まったので、$L$ を最大化するようなパラメータ $\theta$ を求めればよいことになります。ただしイェンセンの不等式を用いた議論でも、結局はKLダイバージェンスを用いた議論が必要になるので先程求めた関係式に戻ることになります。


### パラメータ推定方法

以上の議論から対数尤度を最大化するには、$L$ を最大化するようなパラメータ $\theta$ を求めれば良いことが分かりました。

$$
\ln p(X|\theta) \geq L(q(Z), \theta)
$$

しかしここで今一度、$L$ についての表式を確認してみると、

$$
L(q(Z), \theta) = \sum_{Z} q(Z) \ln \frac{p(X, Z|\theta)}{q(Z)} 
$$

$L$ はパラメータ $\theta$ の関数であると同時に、$q(Z)$ の汎関数であることが分かります。つまり $L$ を最大化するには $\theta$ と $q(Z)$ の組み合わせを見つける必要があるのですが、$q(Z)$ は関数であるので変分法などを用いる必要が出てきます。もう少し簡単に解くために、EMアルゴリズムと呼ばれる枠組みが登場します。EMアルゴリズムでは $\theta$ と $q(Z)$ をそれぞれ逐次的に（別々に）更新することで対数尤度の最大化を行います。徐々に $L$ を大きくしていくようなイメージです。

$$
\ln p(X|\theta) \geq \underset{q,~\theta}{\operatorname{argmax}}~L(q, \theta) \geq ... \geq L(q_i, \theta_i) \geq ... \geq L(q_0, \theta_0)
$$

EM アルゴリズムでは二段階の逐次的最適化手法を採ることで、最尤推定値を見つけることができます。各ステップは以下のように：


- Eステップ：$\underset{q}{\operatorname{argmax}}~L(q, \theta)$
- Mステップ：$\underset{\theta}{\operatorname{argmax}}~L(q, \theta)$

それぞれ $q$ と $\theta$ に対して $L(q, \theta)$ を最大化していきます。


#### Eステップ

$\theta$ を固定して（1回目であれば初期値 $\theta_0$、$i$ 回目であればそれまでに更新された値 $\theta_i$）、$q$ に関して $L$ を最大化します。対数尤度関数の関係式に立ち戻る^[EMアルゴリズムが初見で「？？」となってしまう原因の一つがここにあるかと思っています。最大化しようとしている $L(q,\theta)$ が本当に最大化したい $\ln p(X|\theta)$ に依存しているように見えて、「何やってるんだこれ？」と感じたりします。]と

$$
\begin{aligned}
\ln p(X|\theta_i) &= L(q, \theta_i) + {\rm{KL}} (q||p)  \\
\Leftrightarrow L(q, \theta_i)  &= - {\rm{KL}} (q||p) + \ln p(X|\theta_i) \\
&= - {\rm{KL}} (q||p) + const.
\end{aligned}
$$

いま $\theta_i$ は固定しているので、$\ln p(X|\theta_i)$ は定数であるため

$$
L(q, \theta_i) = - {\rm{KL}} (q||p) + C 
$$

について考えることになります。KLダイバージェンスが $\geq 0$ の量であることを考えると、$L$ は 

$$
{\rm{KL}}(q(Z)||p(Z|X,\theta)) = 0
$$ 

を満たす $q$ の場合に最大になり、

$$
q(Z) = p(Z|X, \theta^i)
$$

となる場合であることが分かります。


#### Mステップ

次に $q$ を固定して $\theta$ に関して $L$ を最大化します。

$$
\begin{aligned}
L(q, \theta) 
&= \sum_{Z} q(Z) \ln \frac{p(X, Z|\theta)}{q(Z)}  \\
&= \sum_{Z} p(Z|X, \theta^{old}) \ln \frac{p(X, Z|\theta)}{p(Z|X, \theta^{old})}  \\
&= \sum_{Z} p(Z|X, \theta^{old}) \ln p(X, Z|\theta) - \sum_{Z} p(Z|X, \theta^{old}) \ln p(Z|X, \theta^{old})  \\
\end{aligned}
$$









## ベイズ的な取り扱い （変分EMアルゴリズム）


ベイズ的に取り扱っているモデル（fully Beysian）に対して、EMアルゴリズムを考えてみます。fully Beysian（完全なベイズ？）では全てのパラメータに対して事前分布を与えるので、これまでに扱っていた $\theta$ も $Z$ の表現の中に取り込まれるため、

$$
\ln p(X) = L(q) + {\rm{KL}} (q || p)
$$

となります。


### 変分法

ベイズ推定とは関係ない（ベイズ推定のための手法ではない）のですが、ここで一旦変分法についてまとめておきます。




### 変分EMアルゴリズム

fully Beysian ですので、汎関数の最適化を解く必要があり、そのため変分法を使用していきます^[汎関数となったのは、パラメータ$\theta$にも事前分布が与えられて、すべて $Z$ に吸収されたためです。]。



# 変分推論 （変分ベイズ）

観測データ $X$ に対して潜在変数などのパラメータ $Z$ を持つ確率モデルを扱う際の主たる目標は

$$
p(Z|X)
$$

で表される事後確率（posterior distribution）を求めることです。ここでモデルは決定論的な（値が一意に決まる）変数を持っているモデルや、完全なベイズモデルである場合などを含むものを想定しています。

現実的には、潜在変数が高次元であることや閉形式の解がなかったりなどと、事後確率を直接計算したり評価したりすることが難しいことが一般的です。そのため近似手法を用いるのですが、（１）決定論的な近似手法と、（２）確率的な（サンプリングを用いた）近似手法の２パターンが存在します。変分ベイズは手法1に相当し、マルコフ連鎖モンテカルロ法などは手法2に相当します。


## 変分推論


全てのパラメータに対して事前分布（prior distrubution）が与えられている完全なベイズモデル（fully Baysian model）を考えます。このモデルも潜在変数を含んでいることもありますが、パラメータと潜在変数はまとめて $Z$ として表記します。


同時分布 $p(X, Z)$ があったときに、目標は事後分布 $p(Z|X)$ を求めることです。最尤法ではパラメータ $\theta$ を（決定論的に）求めていましたが、ベイズ推論ではパラメータの分布を求めることになります。ここで対数尤度についての関係式を再掲します。

$$
\ln p(X) = L(q) + {\rm{KL}} (q || p)
$$

$$
\begin{aligned}
L(q) &= \int q(Z) \ln \left\{ \frac{p(X,Z)}{q(Z)} \right\} dZ \\
\rm{KL}(q||p) &= - \int q(Z) \ln \left\{ \frac{p(Z|X)}{q(Z)} \right\} dZ \\
\end{aligned}
$$

$\theta$ は確率変数となるため、$Z$ として扱われます。（通常の）EMアルゴリズムと同様の手法を採ると、下界である $L(q)$ を $q$ に対して最大化するには、KLダイバージェンスを最小化すればよく、

$$
q(Z) = p(Z|X)
$$

の場合であることが分かります。



### 因子分解




# メモ

- EM アルゴリズムの欠点
  - $\theta$ が点推定であること（最尤法なので）
  - 隠れ変数が1層だけの場合にしか用いれない
- モデルエビデンス
  - モデルエビデンスの推論方法の一つ
- Reparametrization trick
  - サンプリングで誤差逆伝搬できない問題に対処する方法
  - 正規乱数の変換
- 点推定の欠点（=最尤法の欠点）
  - 単峰ではない分布などを表現できず、過学習になる --> ベイズ推定は確率的に予測し、誤差を表示できる
- q(Z)とは？
  - $p(Z|X)$ が求まらないので、$q(Z)$ を準備してそれで評価した代わりにしようというもの



# 資料

- http://chasen.org/~daiti-m/paper/vb-to-vae.pdf
  - VAE は変分ベイズの正統進化のモデル？
  - 通常のNNと異なり、データを簡単に生成できる
    - 通常の autoencoder では latent vector に分布を定義していないので、新しくデータを生成できない
- https://www.ism.ac.jp/~daichi/paper/vb-nlp-tutorial.pdf