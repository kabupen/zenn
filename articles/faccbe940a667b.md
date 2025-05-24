---
title: "iOS 画像認識アプリを作成した①"
emoji: "🤖"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: []
published: false
---

# はじめに

iOS アプリを作りたいと思ってはや幾年、、、。ようやく取り組むための時間ができましたので、実際に作ってみたという技術ブログです。



# 作成物

今回作成したアプリケーションは、画像のアップロードもしくはカメラストリームの撮影から推論を実行するためのアプリケーションです。ソースコードは下記リポジトリにまとめています。

![](https://storage.googleapis.com/zenn-user-upload/f28614ef372a-20250524.jpeg =200x)

- https://github.com/kabupen/ios-vision-app/tree/main

## ContentView.swift

`ContentView` の中で、各サブビューを呼んでいます。

### ImageAreaView

本アプリでは、カメラストリームまたはアップロードした画像を画面中央に配置し、その上に推論結果（バウンディングボックス）を重畳する構成を採用しました。
画像の表示やバウンディングボックスの重畳時には、座標計算にいくつか注意が必要でした。

まず、画像表示領域の設計についてです。親となる View の中にサブビューとして画像エリアを配置していますが、そのサブビューのサイズを取得するために `GeometryReader` を活用しています。


```swift
var body: some View {
    GeometryReader { geometry in
    ...
    var body: some View {
        GeometryReader { proxy in
            let areaWidth = proxy.size.width // サブビューの width を取得
            let areaHeight = proxy.size.height // サブビューの height を取得
            ...
}
```

画像を `.aspectRatio` でアスペクト比を保存したまま親 View に収める場合、親 View と画像とのアスペクト比に差があればその分上下左右に余白が生じることになります。

そのため、親 View のサイズに基づいて拡大縮小や座標計算を行う必要があります。 画像の「実際の表示領域」を求めるには、以下のような計算ロジックを用いました。

```swift
private func calculateImageDisplayRect(
    image: UIImage?,
    areaWidth: CGFloat,
    areaHeight: CGFloat
) -> (aspect: CGFloat, rect: CGRect) {
    guard let image = image else {
        return (1.0, CGRect(x: 0, y: 0, width: areaWidth, height: areaHeight))
    }
    let imageAspect = image.size.width / image.size.height
    let areaAspect = areaWidth / areaHeight
    var displayWidth: CGFloat = areaWidth
    var displayHeight: CGFloat = areaHeight

    if imageAspect > areaAspect {
        displayWidth = areaWidth
        displayHeight = areaWidth / imageAspect
    } else {
        displayHeight = areaHeight
        displayWidth = areaHeight * imageAspect
    }
    let originX = (areaWidth - displayWidth) / 2
    let originY = (areaHeight - displayHeight) / 2
    let rect = CGRect(x: originX, y: originY, width: displayWidth, height: displayHeight)
    return (imageAspect, rect)
}
```

この計算の流れは次のとおりです。

1. 画像とView（表示エリア）のアスペクト比を計算
    - imageAspect … 画像の横幅 ÷ 縦幅
    - areaAspect … View（表示エリア）の横幅 ÷ 縦幅
2. どちらが余白になるか判定
    - 画像が「より横長」なら、Viewの横幅に合わせてfit（上下に余白）
    - 画像が「より縦長」なら、Viewの縦幅に合わせてfit（左右に余白）
3. fit後の画像のサイズ（displayWidth, displayHeight）を計算
    - 画像がViewの中央に来るように、左上座標（originX, originY）を計算
4. (areaWidth - displayWidth) / 2 … 余白を半分ずつ左右・上下に分けて中央寄せ
5. CGRect（＝画像の表示範囲）を返す


このようにして、画像とバウンディングボックスの表示位置のズレを防ぐことができます。



## モデル推論部分

今回は [CoreML-Models](https://github.com/john-rocky/CoreML-Models?tab=readme-ov-file#yolov8) から、YOLOv8s をダウンロードしてきました。そちらのファイルを `ContentView.swift` と同じ階層へ配置します。


```swift
func detectObjectsInImage(_ uiImage: UIImage, completion: @escaping ([DetectionResult]) -> Void) {
    let fixedImage = uiImage.fixedOrientation() // <--- ★ここでExif方向を「正立」に
    // CoreML 用に画像を UIImage --> CIImage に変換
    guard let ciImage = CIImage(image: fixedImage) else { completion([]); return }
    // guard let ciImage = CIImage(image: uiImage) else { completion([]); return }
    print(ciImage.extent.height, ciImage.extent.width)
    // モデル初期化, CoreML モデルを Vision 用にラップ
    // guard let model = try? VNCoreMLModel(for: YOLOv3(configuration: MLModelConfiguration()).model) else { completion([]); return }
    guard let model = try? VNCoreMLModel(for: yolov8s(configuration: MLModelConfiguration()).model) else { completion([]); return }
    
    let request = VNCoreMLRequest(model: model) { req, error in
        guard let results = req.results as? [VNRecognizedObjectObservation] else { completion([]); return }
        
        // デバッグ
        let debugImage = drawBoundingBoxes(on: uiImage, observations: results)
        if let data = debugImage.pngData() {
            let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("debug_bbox.png")
            try? data.write(to: url)
            print("Debug bbox image saved to \(url)")
        }
        
        let mapped = results.map { obj in
            DetectionResult(
                label: obj.labels.first?.identifier ?? "Unknown",
                confidence: obj.confidence,
                rect: obj.boundingBox
            )
        }
        completion(mapped)
    }
    request.imageCropAndScaleOption = .scaleFill
    let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
    DispatchQueue.global().async {
        try? handler.perform([request])
    }
}
```


ユーザーが撮影または選択した画像は、まず `UIImage` というデータ型で取得します。
推論対象とするには `CIImage` 型に変換する必要があります。`CIImage` は CoreML や Vision などの機械学習系APIで主に利用される画像型です。

```swift
guard let ciImage = CIImage(image: image) else { completion([]); return }
```

として `UIImage` から `CIImage` へと変換しました。

次にモデルの初期化についてです。以下の実装で YOLOv8s のモデルの準備を行っています。

```swift
guard let model = try? VNCoreMLModel(for: yolov8s(configuration: MLModelConfiguration()).model) else { completion([]); return }
```

- `yolov8s(configuration: ...).model`
  - CoreMLモデルのインスタンス化を行っています
  - yolov8sはCoreML形式で用意したYOLOv8（小型物体検出モデル）のクラス名で、.mlmodelファイルに合わせて名前は変わります
- `MLModelConfiguration()`
  - モデルロード時の追加設定です
  - 省略可能ですが、カスタム実装やバッチサイズ指定等に使います
- `VNCoreMLModel(for: ...)`
    - CoreMLモデルをVisionフレームワーク ^[画像解析タスクを簡単に実装できるApple公式のフレームワークで、CoreMLと組み合わせることで、自作・学習済みのAIモデルによる画像推論を簡単に行うことができます。] 用にラップしています
    - 物体検出や画像認識用に、Visionの各種リクエストで利用できるように変換しています


作成したモデルを用いた推論処理は、Vision フレームワークの推論リクエストという形式で定義します。

```swift
let request = VNCoreMLRequest(model: model) { req, error in
    guard let results = req.results as? [VNRecognizedObjectObservation] else { completion([]); return }
        let mapped = results.map { obj in
        DetectionResult(
            label: obj.labels.first?.identifier ?? "Unknown",
            confidence: obj.confidence,
            rect: obj.boundingBox
        )
    }
    completion(mapped)
}
```

- VNCoreMLRequest
  - Visionフレームワークの推論リクエストで、物体検出や画像分類などのCoreMLモデルをVision API経由で実行できます
  - クロージャ（コールバック）は推論が終わったときに自動で呼ばれ、req.resultsに推論の生データ（Observationの配列）が入る
- VNRecognizedObjectObservation型
  - 今回は物体検出なので （bbox＋ラベル＋スコア付き）の配列として取得します
  - 型違いや推論失敗時は completion([]) で空配列を返して終了します

実際の推論部分は以下の実装となっています

```swift
let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
DispatchQueue.global().async {
    try? handler.perform([request])
}
```


- 推論リクエストを非同期で実行しています。
  - handler.perform([request]) で、作成した推論リクエスト（request）を実際に画像（ciImage）に対して適用
  - DispatchQueue.global().async でバックグラウンドスレッド上で実行
  - 画像推論は重い処理なので、UIが固まらないように非同期で実行している


# おわりに

今回の実装では、画像のアスペクト比や表示位置を正確に管理し、推論結果をきれいに重畳する部分に特に気を配りました。Vision フレームワークと CoreML を組み合わせることで、iOS アプリ上での画像認識・物体検出がとても簡単かつ強力に実現できます。

 次回は、CoreMLモデルの作成やパフォーマンス最適化についても掘り下げたいと思います。