import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
# import matplotlib.colors as mcolors
# import random
import time

# 시간 측정 시작
start_time = time.time()

# 초기 설정 변경
num_uavs = 10
num_vertices = 50  # 노드 수
frame_number = 100  # 프레임 수
battery_drain = 4.5
battery_charge = 6.5
charging_station = np.array([50, 100], dtype=float)  # 중앙 위치
magma_colors = plt.get_cmap('magma')(np.linspace(0, 1, 100))

# vertices 생성 시 검정 박스 영역을 제외하고 생성
vertices = np.array([np.random.rand(2) * 100 for _ in range(num_vertices) if not (45 <= np.random.rand(2)[0] * 100 <= 55 and 45 <= np.random.rand(2)[1] * 100 <= 55)])

# UAV별 방문할 vertices 분배, vertices 개수를 초과하지 않도록 확인
uav_routes = np.array_split(np.random.permutation(len(vertices)), num_uavs)

# 배터리 상태 및 위치 초기화
battery_levels = np.full(num_uavs, 100.0)
uav_positions = np.array([charging_station] * num_uavs, dtype=float)  # 모든 UAV는 검정 박스 위치에서 시작
uav_colors = [sns.husl_palette(n_colors=num_uavs)[i] for i in range(num_uavs)]  # 각 UAV에 랜덤 색상 할당

# 방문된 vertices 추적
visited = np.zeros(num_vertices, dtype=bool)
visit_time = np.full(num_vertices, -1, dtype=int)  # 방문 시간 초기화
color_indices = np.zeros(num_vertices, dtype=int)  # 색상 인덱스 초기화

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.5, 1]})
plt.subplots_adjust(top=0.85)

# 색상 막대 추가
sm = plt.cm.ScalarMappable(cmap='magma', norm=plt.Normalize(0, 100))
sm.set_array([])
plt.colorbar(sm, ax=ax1, orientation='vertical', label='Time Since Visit', pad=0.1)

def init():
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, num_uavs)
    ax1.add_patch(plt.Rectangle(charging_station - 5, 10, 10, color='black'))  # 검정 박스 그리기

    # 축 레이블 추가
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")

    # Dynamic text including the number of UAVs
    fig.text(0.5, 0.95, f"Number of charging docks = 3, Number of vehicles = {num_uavs}", fontsize=10, ha='center', va='top')

    return fig,

def update(frame):
    ax1.clear()
    ax2.clear()

    # 타이틀 표시
    ax1.set_title(f"Time: {frame} seconds")
    ax2.set_title("Current Battery Status")

    # 검정 박스 그리기
    ax1.add_patch(plt.Rectangle(charging_station - 5, 10, 10, color='black'))

    # Vertices 그리기
    for i, pos in enumerate(vertices):
        if visited[i]:
            # 방문된 시간에 따라 색상 결정
            # 최근 방문 시간을 기준으로 색상을 선택합니다.
            color_index = max(0, color_indices[i] - 3 * frame + 3 * visit_time[i])  # 색상 인덱스 감소
            ax1.scatter(*pos, color=magma_colors[color_index])
        else:
            ax1.scatter(*pos, color='black')

    # 축 레이블 설정
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")

    # 각 UAV 업데이트 및 배터리 상태 그리기
    battery_box_width = 50
    battery_box_height = 1
    for i in range(num_uavs):
        index = frame % len(uav_routes[i])  # 현재 프레임에 대한 인덱스 계산
        current_target = vertices[uav_routes[i][index]]  # 이 부분에서 에러 발생 가능성 감소
        direction = current_target - uav_positions[i]
        distance = np.linalg.norm(direction)

        if distance > 5:
            direction = direction / distance * 5
        else:
            if not visited[uav_routes[i][frame % len(uav_routes[i])]]:
                visited[uav_routes[i][frame % len(uav_routes[i])]] = True
                visit_time[uav_routes[i][frame % len(uav_routes[i])]] = frame
                color_indices[i] = 99  # 최대 색상 인덱스로 설정

        uav_positions[i] += direction

        # UAV 그리기 (색상 및 테두리 추가)
        ax1.add_patch(plt.Polygon(uav_positions[i] + np.array([[0, 2], [-1, -1], [1, -1]]),
                                closed=True,
                                color=uav_colors[i],
                                edgecolor='black',
                                linewidth=2))  # 테두리 두께를 2로 설정

        # 배터리 박스 그리기
        ax2.add_patch(plt.Rectangle((5, i * (battery_box_height + 5)), battery_box_width, battery_box_height, edgecolor='black', facecolor='none'))
        battery_fill = (battery_levels[i] / 100.0) * battery_box_width
        ax2.add_patch(plt.Rectangle((5, i * (battery_box_height + 5)), battery_fill, battery_box_height, facecolor='green' if battery_levels[i] > 50 else 'red'))
        ax2.text(-5, i * (battery_box_height + 5) + battery_box_height / 2, f'UAV {i+1}', verticalalignment='center')
        ax2.text(5 + battery_box_width + 2, i * (battery_box_height + 5) + battery_box_height / 2, f'{battery_levels[i]:.1f}%', verticalalignment='center')

        # 배터리 충전 및 방전 로직
        if not np.allclose(uav_positions[i], charging_station):  # 충전소가 아닐 때 배터리 소모
            battery_levels[i] -= battery_drain
            if battery_levels[i] < 10:  # 배터리 하한선에 도달
                battery_levels[i] = 10  # 최소 배터리 수준 설정
        else:  # 충전소에 있을 때 충전
            battery_levels[i] += battery_charge
            if battery_levels[i] > 100:  # 배터리 상한선 초과
                battery_levels[i] = 100  # 최대 배터리 수준 설정

    # 눈금 제거 및 축 범위 설정
    ax2.set_xlim(0, 70)
    ax2.set_ylim(0, num_uavs * (battery_box_height + 5))
    ax2.axis('off')  # 축과 눈금 제거

ani = animation.FuncAnimation(fig, update, frames=frame_number, init_func=init, repeat=False)
plt.show()

# 시간 측정 종료 및 출력
end_time = time.time()
print("Total Execution Time: {:.2f} seconds".format(end_time - start_time))
