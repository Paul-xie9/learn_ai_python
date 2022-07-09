def string_learn():
    name = 'love lace  '

    # title() 以首字母大写的方式显示每个单词
    print(name.title())
    # 全部大写
    print(name.upper())
    # 全部小写
    print(name.lower())
    # 删除末尾空格
    print(name.rstrip())


def input_learn():
    age = input("How old are you?\n")
    print(age)


def while_learn():
    current_num = 1
    message = ""
    while current_num <= 5:
        print(current_num)
        current_num += 1
        while message != 'q':
            message = input("Enter 'q' to end the program.")


if __name__ == '__main__':
    input_learn()
