import pygame
import sys


# 游戏的所以设置
class Settings:
    def __init__(self):
        # 初始化游戏的设置
        self.screen_width = 1200
        self.screen_height = 800
        self.bg_color = (230, 230, 230)
        # 飞船速度
        self.ship_speed_factor = 1.5


# 创建飞船类
class Ship:
    def __init__(self, settings, screen):
        # 初始化飞船并设置初始化位置
        self.screen = screen
        self.settings = settings
        # 加载飞船图像
        self.image = pygame.image.load('./images/ship.bmp')
        self.rect = self.image.get_rect()
        self.screen_rect = screen.get_rect()
        # 将每艘新飞船放到屏幕底部中央
        self.rect.centerx = self.screen_rect.centerx
        self.rect.bottom = self.screen_rect.bottom
        # 在飞船的属性center中存储小数值
        self.center = float(self.rect.centerx)
        # 移动标志
        self.moving_right = False
        self.moving_left = False

    def update(self):
        if self.moving_right:
            self.rect.centerx += self.settings.ship_speed_factor
        if self.moving_left:
            self.rect.centerx -= self.settings.ship_speed_factor
        self.rect.centerx = self.center

    def blitme(self):
        # 在指定位置绘制飞船
        self.screen.blit(self.image, self.rect)


# 监视键盘和鼠标事件
def check_events(ship):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            print('event.key', event.key)
            if event.key == pygame.K_RIGHT:
                ship.moving_right = True  # 右移
            elif event.key == pygame.K_LEFT:
                ship.moving_left = True  # 左移
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_RIGHT:
                ship.moving_right = False
            elif event.key == pygame.K_LEFT:
                ship.moving_left = False


# 更新屏幕
def update_screen(bg_color, screen, ship):
    # 每次循环时都会重绘屏幕
    screen.fill(bg_color)
    ship.blitme()
    # 让最近绘制的屏幕可见
    pygame.display.flip()


def run_game():
    # 初始化游戏并创建一个屏幕
    pygame.init()
    settings = Settings()
    screen_size = (settings.screen_width, settings.screen_height)  # 屏幕尺寸
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Alien Invasion")
    # 设置背景颜色
    bg_color = settings.bg_color
    # 创建一艘飞船
    ship = Ship(settings, screen)
    # 开始游戏的主循环
    while True:
        # 监视键盘和鼠标事件
        check_events(ship)
        ship.update()
        # 每次循环时都会重绘屏幕
        update_screen(bg_color, screen, ship)


if __name__ == '__main__':
    run_game()
