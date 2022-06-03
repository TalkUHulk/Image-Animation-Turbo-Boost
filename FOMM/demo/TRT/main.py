from flask import Flask
from api import fom_bp

app = Flask(__name__)
app.register_blueprint(fom_bp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12581)


