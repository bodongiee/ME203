#!/usr/bin/env python3
# steering_control_diagram_mpl.py
# OpenCV 좌표계에서 correction_error, servo_angle, angle_correction 관계 시각화 (matplotlib 버전)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle, Wedge
import numpy as np
import math

def create_diagram():
    """제어 관계 다이어그램 생성"""

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#2a2a2a')

    # ============ 1. OpenCV 좌표계 ============
    ax1 = plt.subplot(3, 2, 1)
    ax1.set_xlim(-0.5, 5)
    ax1.set_ylim(-0.5, 5)
    ax1.set_aspect('equal')
    ax1.set_facecolor('#3a3a3a')

    # X축 (오른쪽)
    ax1.arrow(0, 0, 3, 0, head_width=0.3, head_length=0.3, fc='red', ec='red', linewidth=2)
    ax1.text(3.5, 0, 'X (right)', color='red', fontsize=12, fontweight='bold')

    # Y축 (아래)
    ax1.arrow(0, 0, 0, 3, head_width=0.3, head_length=0.3, fc='lime', ec='lime', linewidth=2)
    ax1.text(0.2, 3.5, 'Y (down)', color='lime', fontsize=12, fontweight='bold')

    ax1.set_title('OpenCV Coordinate System', color='white', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    # ============ 2. 카메라 뷰 시뮬레이션 ============
    ax2 = plt.subplot(3, 2, (2, 4))
    ax2.set_xlim(0, 240)
    ax2.set_ylim(240, 0)  # Y축 반전 (OpenCV 좌표계)
    ax2.set_aspect('equal')
    ax2.set_facecolor('#2a2a2a')

    # 카메라 프레임
    frame = Rectangle((0, 0), 240, 240, linewidth=2, edgecolor='gray', facecolor='#1a1a1a')
    ax2.add_patch(frame)

    ax2.set_title('Camera View (240×240) - Line Detection', color='white', fontsize=14, fontweight='bold', pad=10)

    # 화면 중심선 (목표)
    center_x = 120
    ax2.plot([center_x, center_x], [0, 240], 'c--', linewidth=1.5, label='Target center')
    ax2.text(center_x + 5, 20, 'Center\n(target)', color='cyan', fontsize=9)

    # 검출된 라인 (예: 오른쪽으로 치우침)
    line_center_x_norm = 0.7  # 0~1 정규화
    line_x = int(240 * line_center_x_norm)
    line_angle = 110.0  # 도 단위

    # 라인 그리기 (타원형으로 각도 표현)
    angle_rad = math.radians(line_angle)
    line_points_x = []
    line_points_y = []

    for y in range(40, 200, 15):
        offset = (y - 40) / 160 * 30
        x = line_x + offset * math.cos(angle_rad)
        line_points_x.append(x)
        line_points_y.append(y)

    ax2.plot(line_points_x, line_points_y, 'wo', markersize=10, label='Detected line')
    ax2.plot(line_points_x, line_points_y, 'w-', linewidth=2, alpha=0.5)

    # 라인 중심 표시
    ax2.plot(line_x, 120, 'ro', markersize=8, label='Line center')
    ax2.text(line_x + 10, 120, f'Line X={line_x}px\n(norm={line_center_x_norm})',
             color='red', fontsize=9, bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # center_error 화살표
    arrow_y = 220
    ax2.annotate('', xy=(line_x, arrow_y), xytext=(center_x, arrow_y),
                arrowprops=dict(arrowstyle='<->', color='orange', lw=2))

    center_error = (line_center_x_norm - 0.5) * 2  # -1 ~ 1
    ax2.text((center_x + line_x) / 2, arrow_y - 10,
             f'center_error = {center_error:.2f}',
             color='orange', fontsize=11, fontweight='bold',
             ha='center', bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

    # 라인 각도 표시
    ax2.text(line_x + 50, 80, f'line_angle\n= {line_angle:.1f}°',
             color='yellow', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

    ax2.legend(loc='upper left', fontsize=9, facecolor='#3a3a3a', edgecolor='gray', labelcolor='white')
    ax2.set_xlabel('X (pixels)', color='white', fontsize=10)
    ax2.set_ylabel('Y (pixels)', color='white', fontsize=10)
    ax2.tick_params(colors='white')

    # ============ 3. 제어 흐름 다이어그램 ============
    ax3 = plt.subplot(3, 2, 5)
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 6)
    ax3.set_aspect('equal')
    ax3.set_facecolor('#2a2a2a')
    ax3.axis('off')

    # Step 1: center_error → PID
    step1 = Rectangle((0.5, 4), 2.5, 1.2, linewidth=2, edgecolor='#6495ED', facecolor='#1e3a5f')
    ax3.add_patch(step1)
    ax3.text(1.75, 4.8, '1. PID Controller', ha='center', va='center',
             color='#6495ED', fontsize=10, fontweight='bold')
    ax3.text(1.75, 4.4, f'Input: error={center_error:.2f}', ha='center', va='center',
             color='white', fontsize=8)

    # PID 출력
    kp, ki, kd = 1.8, 0.02, 1.0
    correction = kp * center_error
    ax3.arrow(3.2, 4.6, 0.8, 0, head_width=0.2, head_length=0.15, fc='white', ec='white')
    ax3.text(3.6, 5.1, f'correction\n={correction:.2f}', ha='center',
             color='white', fontsize=8, bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

    # Step 2: Angle correction
    step2 = Rectangle((4.2, 4), 2.5, 1.2, linewidth=2, edgecolor='#FF8C00', facecolor='#5f3a1e')
    ax3.add_patch(step2)
    ax3.text(5.45, 4.8, '2. Angle Correction', ha='center', va='center',
             color='#FF8C00', fontsize=10, fontweight='bold')

    angle_correction = (line_angle - 90) * 0.2
    ax3.text(5.45, 4.4, f'(angle-90)×0.2', ha='center', va='center',
             color='white', fontsize=8)
    ax3.text(5.45, 4.1, f'={angle_correction:.2f}', ha='center', va='center',
             color='white', fontsize=8)

    ax3.arrow(6.9, 4.6, 0.8, 0, head_width=0.2, head_length=0.15, fc='white', ec='white')

    # Step 3: Final servo angle
    step3 = Rectangle((7.9, 3.7), 2, 1.8, linewidth=3, edgecolor='#32CD32', facecolor='#1e5f1e')
    ax3.add_patch(step3)
    ax3.text(8.9, 5.2, '3. Servo Angle', ha='center', va='center',
             color='#32CD32', fontsize=11, fontweight='bold')

    servo_angle = 90 - (correction * 30) - angle_correction
    servo_angle = max(45, min(135, servo_angle))

    ax3.text(8.9, 4.7, f'90-(corr×30)-angle_corr', ha='center', va='center',
             color='white', fontsize=7)
    ax3.text(8.9, 4.3, f'=90-{correction*30:.1f}-{angle_correction:.1f}', ha='center', va='center',
             color='white', fontsize=7)
    ax3.text(8.9, 3.9, f'= {servo_angle:.1f}°', ha='center', va='center',
             color='lime', fontsize=12, fontweight='bold')

    # 공식 설명
    ax3.text(5, 2.5, 'Key Formulas:', color='white', fontsize=11, fontweight='bold')
    formulas = [
        '• center_error = (line_x - 0.5) × 2    [-1 to +1]',
        '• PID: correction = Kp×error + Ki×∫error + Kd×d(error)/dt',
        '• angle_correction = (line_angle - 90°) × 0.2',
        '• servo_angle = 90° - (correction×30) - angle_correction',
        '• Final range: [45°, 135°]',
    ]

    for i, formula in enumerate(formulas):
        ax3.text(0.5, 2.0 - i*0.35, formula, color='lightgray', fontsize=8, family='monospace')

    ax3.set_title('Control Flow', color='white', fontsize=14, fontweight='bold', pad=10)

    # ============ 4. 서보 각도 시각화 (Top View) ============
    ax4 = plt.subplot(3, 2, 3)
    ax4.set_xlim(-3, 3)
    ax4.set_ylim(-3, 3)
    ax4.set_aspect('equal')
    ax4.set_facecolor('#2a2a2a')
    ax4.axis('off')

    # 차량 몸체
    car = Rectangle((-0.4, -0.6), 0.8, 1.2, linewidth=2, edgecolor='white', facecolor='gray')
    ax4.add_patch(car)
    ax4.text(0, 0, 'Robot\n(Top View)', ha='center', va='center', color='black', fontsize=9, fontweight='bold')

    # 서보 각도 범위 표시
    radius = 2.5

    # 45도 (최대 왼쪽)
    angle_45_rad = math.radians(90 - 45)  # matplotlib 좌표계 변환
    x1 = radius * math.cos(angle_45_rad)
    y1 = radius * math.sin(angle_45_rad)
    ax4.plot([0, x1], [0, y1], 'gray', linewidth=1, linestyle='--')
    ax4.text(x1 - 0.5, y1 + 0.3, '45°\n(max left)', ha='right', color='gray', fontsize=8)

    # 90도 (직진)
    angle_90_rad = math.radians(90 - 90)
    x2 = radius * math.cos(angle_90_rad)
    y2 = radius * math.sin(angle_90_rad)
    ax4.plot([0, x2], [0, y2], 'cyan', linewidth=2, linestyle='--')
    ax4.text(x2 + 0.2, y2 + 0.5, '90° (straight)', ha='left', color='cyan', fontsize=9, fontweight='bold')

    # 135도 (최대 오른쪽)
    angle_135_rad = math.radians(90 - 135)
    x3 = radius * math.cos(angle_135_rad)
    y3 = radius * math.sin(angle_135_rad)
    ax4.plot([0, x3], [0, y3], 'gray', linewidth=1, linestyle='--')
    ax4.text(x3 + 0.5, y3 + 0.3, '135°\n(max right)', ha='left', color='gray', fontsize=8)

    # 각도 범위 표시 (부채꼴)
    wedge = Wedge((0, 0), radius, 90-135, 90-45, facecolor='yellow', alpha=0.1, edgecolor='yellow', linewidth=1)
    ax4.add_patch(wedge)

    # 현재 서보 각도
    angle_current_rad = math.radians(90 - servo_angle)
    x_current = radius * math.cos(angle_current_rad)
    y_current = radius * math.sin(angle_current_rad)
    ax4.arrow(0, 0, x_current, y_current, head_width=0.3, head_length=0.3,
              fc='lime', ec='lime', linewidth=3)
    ax4.text(x_current * 0.7, y_current * 0.7 + 0.5, f'Current:\n{servo_angle:.1f}°',
             ha='center', color='lime', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

    ax4.set_title('Servo Angle (Top View)', color='white', fontsize=14, fontweight='bold', pad=10)

    # ============ 5. 설명 텍스트 ============
    ax5 = plt.subplot(3, 2, 6)
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 10)
    ax5.set_facecolor('#2a2a2a')
    ax5.axis('off')

    explanation = [
        'Control Logic Summary:',
        '',
        '1. Line Detection:',
        '   • Segmentation model outputs binary mask',
        '   • Extract line center position (0~1 normalized)',
        '   • Extract line angle using ellipse fitting',
        '',
        '2. Error Calculation:',
        '   • center_error = (line_x - 0.5) × 2',
        '   • Positive: line is RIGHT of center → turn LEFT',
        '   • Negative: line is LEFT of center → turn RIGHT',
        '',
        '3. PID Control:',
        '   • Kp=1.8: Proportional gain (main correction)',
        '   • Ki=0.02: Integral (eliminates steady-state error)',
        '   • Kd=1.0: Derivative (dampens oscillation)',
        '',
        '4. Angle Compensation:',
        '   • If line tilts right (>90°) → steer slightly right',
        '   • If line tilts left (<90°) → steer slightly left',
        '',
        '5. Final Output:',
        '   • servo_angle = 90 - (correction×30) - angle_correction',
        '   • Clamped to [45°, 135°] for safety',
        '   • Lower angle = turn left, Higher = turn right',
    ]

    y_pos = 9.5
    for line in explanation:
        if line.startswith('Control Logic') or line.endswith(':'):
            ax5.text(0.5, y_pos, line, color='yellow', fontsize=10, fontweight='bold')
        elif line.startswith('   •'):
            ax5.text(1, y_pos, line, color='lightgray', fontsize=8, family='monospace')
        else:
            ax5.text(0.5, y_pos, line, color='white', fontsize=9)
        y_pos -= 0.38

    # 전체 제목
    fig.suptitle('Line Tracking Control System - Steering Angle Calculation\nOpenCV Coordinate System',
                 color='cyan', fontsize=16, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    return fig

if __name__ == '__main__':
    print("Generating steering control diagram with matplotlib...")

    fig = create_diagram()

    # 저장
    output_path = "./steering_control_diagram.png"
    fig.savefig(output_path, dpi=150, facecolor='#2a2a2a', edgecolor='none')

    print(f"✓ Diagram saved: {output_path}")
    print("Opening image...")

    plt.show()
