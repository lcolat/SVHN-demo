from flask import Flask,render_template,request
import rand_testing_img
import test_model

app = Flask(__name__)

@app.route('/')
def index():
	images = rand_testing_img.main('./dataset/test_32x32.mat')
	models = [
		{
			'value' : 'conv2d_SVHN_grayscale',
			'choix' : 'Modele 1'
		},
		{
			'value': 'res_net_SVHN_grayscale',
			'choix' : 'Modele 2'
		}
	]

	imgs = [
		{
			'value': (images[0][2], images[0][0]),
			'name' : 'image 1'
		},
		{
			'value': (images[1][2], images[1][0]),
			'name' : 'image 2'
		},
		{
			'value': (images[2][2], images[2][0]),
			'name' : 'image 3'
		},
		{
			'value': (images[3][2], images[3][0]),
			'name' : 'image 4'
		},
		{
			'value': (images[4][2], images[4][0]),
			'name' : 'image 5'
		},
		{
			'value': (images[5][2], images[5][0]),
			'name' : 'image 6'
		},
	]
	return render_template('index.html', models=models, imgs=imgs, images=images)

@app.route('/start', methods=['GET', 'POST'])
def start():
	if request.method == 'POST':
		resultM = request.form.get('select1')
		resultI = eval(request.form.get('select2'))
		expected_result, result, percentage = test_model.main('./models/' + resultM, './dataset/test_32x32.mat', int(resultI[0]))
		return render_template("start.html", expected_result=expected_result, result=result, percentage=percentage, image=resultI[1])

