from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from Medicine import Medicine, MedicineCategory

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///your_database.db'  # 示例数据库连接
db = SQLAlchemy(app)


# 获取所有药品信息
@app.route('/api/products', methods=['GET'])
def get_medicines():
    medicines = Medicine.query.all()
    # 转换为JSON格式返回
    return jsonify([medicine.to_dict() for medicine in medicines])


# 添加药品信息
@app.route('/api/products', methods=['POST'])
def add_medicine():
    data = request.get_json()
    # 数据验证等逻辑...
    new_medicine = Medicine(**data)
    db.session.add(new_medicine)
    db.session.commit()
    return jsonify({'message': 'Medicine added successfully'}), 201


# ... 其他操作如更新、删除药品信息的接口可以类似设计

if __name__ == '__main__':
    app.run(debug=True)
