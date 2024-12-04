import sys
import random
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QGridLayout, QWidget, QLabel, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont


class Minesweeper(QMainWindow):
    def __init__(self, rows=10, cols=10, mines=10):
        super().__init__()
        self.setWindowTitle("扫雷游戏")
        self.setFixedSize(400, 450)

        self.rows = rows
        self.cols = cols
        self.mines = mines
        self.flags_left = mines
        self.is_game_over = False

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)
        self.time_elapsed = 0

        self.initUI()
        self.start_game()

    def initUI(self):
        """初始化用户界面"""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QGridLayout()
        self.layout.setSpacing(0)
        self.central_widget.setLayout(self.layout)

        # 创建网格按钮
        self.buttons = {}
        for row in range(self.rows):
            for col in range(self.cols):
                button = QPushButton("")
                button.setFixedSize(40, 40)

                # 设置字体
                button.setFont(QFont("Arial", 14))

                button.clicked.connect(lambda _, r=row, c=col: self.reveal_cell(r, c))
                button.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
                button.customContextMenuRequested.connect(lambda _, r=row, c=col: self.flag_cell(r, c))
                self.layout.addWidget(button, row, col)
                self.buttons[(row, col)] = button

        # 标题栏显示计时器与标记剩余地雷
        self.statusBar = self.statusBar()
        self.timer_label = QLabel("时间: 0")
        self.flags_label = QLabel(f"剩余旗帜: {self.flags_left}")
        self.statusBar.addPermanentWidget(self.timer_label)
        self.statusBar.addPermanentWidget(self.flags_label)

    def start_game(self):
        """开始游戏，初始化地雷和网格"""
        self.board = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.is_revealed = [[False for _ in range(self.cols)] for _ in range(self.rows)]
        self.is_flagged = [[False for _ in range(self.cols)] for _ in range(self.rows)]

        self.place_mines()
        self.calculate_adjacent_mines()

        # 启动计时器
        self.timer.start(1000)

    def update_timer(self):
        """更新计时器"""
        self.time_elapsed += 1
        self.timer_label.setText(f"时间: {self.time_elapsed}")

    def place_mines(self):
        """随机放置地雷"""
        mines_placed = 0
        while mines_placed < self.mines:
            row = random.randint(0, self.rows - 1)
            col = random.randint(0, self.cols - 1)
            if self.board[row][col] != -1:
                self.board[row][col] = -1
                mines_placed += 1

    def calculate_adjacent_mines(self):
        """计算每个格子周围的地雷数量"""
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for row in range(self.rows):
            for col in range(self.cols):
                if self.board[row][col] == -1:
                    continue
                mine_count = 0
                for dr, dc in directions:
                    r, c = row + dr, col + dc
                    if 0 <= r < self.rows and 0 <= c < self.cols and self.board[r][c] == -1:
                        mine_count += 1
                self.board[row][col] = mine_count

    def reveal_cell(self, row, col):
        """点击格子进行翻开"""
        if self.is_game_over or self.is_flagged[row][col]:
            return

        button = self.buttons[(row, col)]
        button.setEnabled(False)
        self.is_revealed[row][col] = True

        if self.board[row][col] == -1:
            button.setText("💣")
            button.setStyleSheet("color: red;")
            self.game_over(False)
        else:
            button.setText(str(self.board[row][col]) if self.board[row][col] > 0 else "")
            button.setStyleSheet("color: black;")
            if self.board[row][col] == 0:
                self.reveal_adjacent_cells(row, col)

        if self.check_win():
            self.game_over(True)

    def reveal_adjacent_cells(self, row, col):
        """递归揭示周围空白格子"""
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < self.rows and 0 <= c < self.cols and not self.is_revealed[r][c]:
                self.reveal_cell(r, c)

    def flag_cell(self, row, col):
        """右键单击标记地雷"""
        if self.is_game_over or self.is_revealed[row][col]:
            return

        button = self.buttons[(row, col)]
        if not self.is_flagged[row][col]:
            button.setText("🚩")
            self.is_flagged[row][col] = True
            self.flags_left -= 1
        else:
            button.setText("")
            self.is_flagged[row][col] = False
            self.flags_left += 1

        self.flags_label.setText(f"剩余旗帜: {self.flags_left}")
        if self.check_win():
            self.game_over(True)

    def check_win(self):
        """检查玩家是否赢得游戏"""
        for row in range(self.rows):
            for col in range(self.cols):
                if self.board[row][col] == -1 and not self.is_flagged[row][col]:
                    return False
        return True

    def game_over(self, won):
        """游戏结束"""
        self.timer.stop()
        self.is_game_over = True
        if won:
            QMessageBox.information(self, "胜利！", "恭喜你！你成功扫除了所有地雷！")
        else:
            QMessageBox.critical(self, "失败！", "你踩到了地雷，游戏结束！")
        self.reset_game()

    def reset_game(self):
        """重置游戏"""
        self.is_game_over = False
        self.time_elapsed = 0
        self.flags_left = self.mines
        self.timer_label.setText("时间: 0")
        self.flags_label.setText(f"剩余旗帜: {self.flags_left}")
        self.start_game()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Minesweeper()
    window.show()
    sys.exit(app.exec())
