import tkinter as tk
from tkinter import ttk


def create_gui():
    # 创建主窗口
    root = tk.Tk()
    root.title("自定义GUI界面")
    root.geometry("400x300")  # 设置窗口大小

    # 第一部分：四个选择（单选按钮）
    frame1 = tk.Frame(root)
    frame1.pack(pady=10)

    var1 = tk.IntVar()
    var1.set(0)  # 初始化选中状态
    choices1 = [("选择1", 1), ("选择2", 2), ("选择3", 3), ("选择4", 4)]
    for text, value in choices1:
        tk.Radiobutton(frame1, text=text, variable=var1, value=value).pack(side=tk.LEFT, padx=5)

        # 第二部分：两个输入框和对应的文字解释
    frame2 = tk.Frame(root)
    frame2.pack(pady=10)

    tk.Label(frame2, text="输入框1:").pack(side=tk.LEFT)
    entry1 = tk.Entry(frame2)
    entry1.pack(side=tk.LEFT)

    tk.Label(frame2, text="这是输入框1的解释").pack(side=tk.LEFT)

    tk.Label(frame2, text="输入框2:").pack(side=tk.LEFT)
    entry2 = tk.Entry(frame2)
    entry2.pack(side=tk.LEFT)

    tk.Label(frame2, text="这是输入框2的解释").pack(side=tk.LEFT)

    # 第三部分：下拉框包含三个选项
    frame3 = tk.Frame(root)
    frame3.pack(pady=10)

    var2 = tk.StringVar()
    var2.set("选项1")  # 设置默认选项
    combobox = ttk.Combobox(frame3, textvariable=var2)
    combobox['values'] = ("选项1", "选项2", "选项3")
    combobox.pack()

    # 运行主循环
    root.mainloop()


if __name__ == "__main__":
    create_gui()