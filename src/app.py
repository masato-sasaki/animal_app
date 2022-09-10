# 必要なモジュールのインポート
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import io
import base64
from PIL import Image
import torch
from animal import transform, Net #animal.pyから前処理とネットワークの定義を読み込み

# 学習済みモデルをもとに推論する
def predict(img):
    # ネットワークの準備
    net = Net().cpu().eval()
    # # 学習済みモデルの重み（dog_cat.pt）を読み込み
    net.load_state_dict(torch.load('./dog_cat.pt', map_location=torch.device('cpu')))
    #　データの前処理
    img = transform(img)
    img =img.unsqueeze(0) # 1次元増やす
    #　推論
    y = torch.argmax(net(img), dim=1).cpu().detach().numpy()
    return y

#　推論したラベルから犬か猫かを返す
def getName(label):
    if label==0:
        return "猫"
    elif label==1:
        return "犬"

# python オブジェクト読み込みの設定とセッション情報の暗号化
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '8787fueyu293748huhsjekjuh'

# 画像のアップロード崎のディレクトリ
UPLOAD_FOLDER = './src/static/img'
# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'jpeg'])

#　拡張子が適切かどうかをチェック
def allwed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# URL にアクセスがあった場合の挙動の設定
@app.route('/', methods = ['GET', 'POST'])
def predicts():

    animalName = ''
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        # データの取り出し
        file = request.files['file']
        # ファイル名がなかった時の処理
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        # ファイルのチェック
        if file and allwed_file(file.filename):
            # 不正なファイル名でないことを確認して取得
            filename = secure_filename(file.filename)
            
            #　画像ファイルに対する処理
            #　画像書き込み用バッファを確保
            buf = io.BytesIO()
            image = Image.open(file)
            #　画像データをバッファに書き込む
            image.save(buf, 'png')
            #　バイナリデータをbase64でエンコードしてutf-8でデコード
            base64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            #　HTML側のsrcの記述に合わせるために付帯情報付与する
            base64_data = "data:image/png;base64,{}".format(base64_str)

            # 入力された画像に対して推論
            pred = predict(image)
            animalName = getName(pred)
            return render_template('result.html', animalName=animalName, img=base64_data)

    # GET 　メソッドの定義
    elif request.method == 'GET':
        return render_template('index.html')


# アプリケーションの実行の定義
if __name__ == "__main__":
    app.run()