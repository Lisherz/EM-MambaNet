import torch
import numpy as np
import matplotlib.pyplot as plt
from models.encoders.dual_vmamba import vssm_tiny
from dataloader.dataloader import get_train_loader


def run_edge_diagnosis():
    """运行边缘诊断"""

    # 1. 创建诊断模型
    model = vssm_tiny(
        enable_detailed_diagnosis=True,
        diagnosis_interval=1,  # 每个batch都诊断
        use_edge_modulation=True,
        modulation_stage=2,
        adaptive_edge_strength=True
    )
    model.cuda()
    model.train()

    # 2. 获取测试数据
    test_loader = get_test_loader(batch_size=2, num_samples=10)

    # 3. 运行诊断
    print("=" * 70)
    print("🔍 Running Edge Injection Diagnosis")
    print("=" * 70)

    for batch_idx, batch in enumerate(test_loader):
        print(f"\n📦 Batch {batch_idx + 1}/10")

        As = batch['A'].cuda()
        Bs = batch['B'].cuda()

        # 前向传播（会收集诊断数据）
        with torch.no_grad():
            logits, _ = model(As, Bs)

        # 打印诊断报告
        model.print_detailed_diagnosis(iteration=f"Batch{batch_idx + 1}")

        # 只运行少量batch进行诊断
        if batch_idx >= 2:
            break

    # 4. 生成诊断总结报告
    print("\n" + "=" * 70)
    print("📋 DIAGNOSIS SUMMARY REPORT")
    print("=" * 70)

    report = model.get_diagnosis_report(clear_data=False)

    if report:
        # 分析常见问题
        common_issues = analyze_common_issues(report)

        print("\n🔎 Common Edge Injection Issues Detected:")
        for issue, details in common_issues.items():
            print(f"\n  {issue}:")
            for detail in details:
                print(f"    • {detail}")

    # 5. 可视化边缘图
    visualize_edge_maps(model, test_loader)

    return report


def analyze_common_issues(report):
    """分析常见问题"""
    issues = {}

    # 检查边缘质量
    if 'edge_quality' in report.get('summary', {}):
        eq = report['summary']['edge_quality']

        if eq['mean'] < 0.05:
            issues.setdefault('Poor Edge Detection', []).append(
                f"Edge map mean too low ({eq['mean']:.4f})"
            )

        if eq['sparsity'] > 0.9:
            issues.setdefault('Sparse Edges', []).append(
                f"Edge map too sparse ({eq['sparsity']:.3f})"
            )

    # 检查调制影响
    if 'modulation_impact' in report.get('summary', {}):
        mi = report['summary']['modulation_impact']

        if mi['mean_change'] > 0.5:
            issues.setdefault('Excessive Modulation', []).append(
                f"Modulation change too large ({mi['mean_change']:.4f})"
            )

        if mi['correlation'] < 0.3:
            issues.setdefault('Feature Corruption', []).append(
                f"Low feature correlation after modulation ({mi['correlation']:.3f})"
            )

    # 检查强度参数
    if 'strength_params' in report.get('summary', {}):
        sp = report['summary']['strength_params']

        if len(sp) >= 3 and sp[2] > 1.5:  # stage2强度
            issues.setdefault('High Injection Strength', []).append(
                f"Stage 2 injection strength too high ({sp[2]:.3f})"
            )

    return issues


def visualize_edge_maps(model, dataloader):
    """可视化边缘图"""
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 3:  # 只看3个样本
                break

            As = batch['A'].cuda()
            Bs = batch['B'].cuda()

            # 获取SAR特征
            outs_B = model.vssm(Bs)

            # 提取边缘图
            edge_map = model.extract_edge_maps(outs_B[2])

            # 转换为numpy
            edge_np = edge_map[0, 0].cpu().numpy()

            # 可视化
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(As[0].permute(1, 2, 0).cpu().numpy())
            plt.title('Optical Image')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(Bs[0, 0].cpu().numpy(), cmap='gray')
            plt.title('SAR Image')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(edge_np, cmap='hot')
            plt.title('Extracted Edge Map')
            plt.colorbar()
            plt.axis('off')

            plt.suptitle(f'Sample {batch_idx + 1}')
            plt.tight_layout()
            plt.show()

            # 打印边缘统计
            print(f"\n📊 Sample {batch_idx + 1} Edge Statistics:")
            print(f"  Min: {edge_np.min():.4f}, Max: {edge_np.max():.4f}")
            print(f"  Mean: {edge_np.mean():.4f}, Std: {edge_np.std():.4f}")

    model.train()


if __name__ == '__main__':
    report = run_edge_diagnosis()

    # 保存诊断报告
    if report:
        import json

        with open('edge_diagnosis_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        print("\n✅ Diagnosis report saved to 'edge_diagnosis_report.json'")