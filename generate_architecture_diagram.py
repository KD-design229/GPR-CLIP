#!/usr/bin/env python3
"""
生成架构图脚本 - 使用graphviz创建流程图和架构图
Generate architecture diagrams using graphviz
"""

import os

def create_graphviz_diagrams():
    """创建多个架构图的DOT源码"""
    
    # 1. 整体系统架构图
    system_architecture = """
digraph SystemArchitecture {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor=lightblue, fontname="Arial"];
    
    subgraph cluster_0 {
        label="Server Layer";
        style=filled;
        fillcolor=lightgray;
        
        ServerBase [label="ServerBase\\n(基础服务器类)", fillcolor=lightgreen];
        FedDWA [label="FedDWA\\n(动态权重聚合)"];
        FedAvg [label="FedAvg\\n(联邦平均)"];
        FedProx [label="FedProx\\n(近端正则化)"];
    }
    
    subgraph cluster_1 {
        label="Client Layer";
        style=filled;
        fillcolor=lightgray;
        
        ClientBase [label="ClientBase\\n(基础客户端类)", fillcolor=lightgreen];
        ClientDWA [label="ClientFedDWA\\n(两步训练)"];
        ClientAvg [label="ClientFedAvg\\n(标准训练)"];
    }
    
    subgraph cluster_2 {
        label="Model Layer";
        style=filled;
        fillcolor=lightgray;
        
        CNN [label="CIFAR10Model\\n(基础CNN)"];
        ResNet [label="ResNet18\\n(残差网络)"];
        MobileViT [label="MobileViT\\n(Vision Transformer)"];
        FedCLIP [label="FedCLIP\\n(多模态CLIP)"];
        GPRFed [label="GPR-FedSense\\n(探地雷达)"];
    }
    
    ServerBase -> FedDWA [label="继承"];
    ServerBase -> FedAvg [label="继承"];
    ServerBase -> FedProx [label="继承"];
    
    ClientBase -> ClientDWA [label="继承"];
    ClientBase -> ClientAvg [label="继承"];
    
    FedDWA -> ClientDWA [label="交互", style=dashed, color=red];
    FedAvg -> ClientAvg [label="交互", style=dashed, color=red];
    
    ClientDWA -> CNN [label="使用", style=dashed, color=blue];
    ClientDWA -> ResNet [label="使用", style=dashed, color=blue];
    ClientDWA -> MobileViT [label="使用", style=dashed, color=blue];
    ClientDWA -> FedCLIP [label="使用", style=dashed, color=blue];
    ClientDWA -> GPRFed [label="使用", style=dashed, color=blue];
}
"""
    
    # 2. FedDWA算法流程图
    feddwa_workflow = """
digraph FedDWAWorkflow {
    rankdir=TB;
    node [shape=box, style="rounded,filled", fillcolor=lightblue, fontname="Arial"];
    
    Start [label="开始\\n(Round t)", shape=ellipse, fillcolor=lightgreen];
    SelectClient [label="Server选择K个客户端"];
    SendModel [label="Server发送模型 w_i^{t-1}"];
    
    subgraph cluster_client {
        label="Client i 本地训练";
        style=filled;
        fillcolor=lightyellow;
        
        Train1 [label="训练E个epoch\\n→ 得到 w_i^t"];
        Train2 [label="额外训练1步\\n→ 得到 w_i^{t+1}"];
        Upload [label="上传 (w_i^t, w_i^{t+1})"];
        
        Train1 -> Train2;
        Train2 -> Upload;
    }
    
    CalcWeight [label="Server计算权重矩阵\\nW[j,i] ∝ 1/||w_i^{t+1} - w_j^t||²"];
    TopK [label="Top-K选择 + 归一化"];
    Aggregate [label="个性化聚合\\nw_i^{new} = Σ W[j,i]*w_j^t"];
    SendNew [label="发送个性化模型 w_i^{new}"];
    
    Decision [label="达到T轮?", shape=diamond, fillcolor=lightcoral];
    End [label="结束", shape=ellipse, fillcolor=lightcoral];
    
    Start -> SelectClient;
    SelectClient -> SendModel;
    SendModel -> Train1;
    Upload -> CalcWeight;
    CalcWeight -> TopK;
    TopK -> Aggregate;
    Aggregate -> SendNew;
    SendNew -> Decision;
    Decision -> SelectClient [label="否"];
    Decision -> End [label="是"];
}
"""
    
    # 3. FedCLIP架构图
    fedclip_architecture = """
digraph FedCLIPArchitecture {
    rankdir=LR;
    node [shape=box, style="rounded,filled", fillcolor=lightblue, fontname="Arial"];
    
    Input [label="输入图像\\n224×224×3", shape=parallelogram, fillcolor=lightgreen];
    
    subgraph cluster_clip {
        label="CLIP Backbone (冻结)";
        style=filled;
        fillcolor=lightgray;
        
        ImageEncoder [label="Image Encoder\\nViT-B/32"];
        ImageFeatures [label="图像特征\\n512D"];
        
        ImageEncoder -> ImageFeatures;
    }
    
    subgraph cluster_adapter {
        label="Trainable Adapter";
        style=filled;
        fillcolor=lightyellow;
        
        MLP1 [label="MaskedMLP\\n(512→512)"];
        BN [label="BatchNorm1d"];
        ReLU [label="ReLU"];
        MLP2 [label="MaskedMLP\\n(512→512)"];
        Softmax [label="Softmax"];
        
        MLP1 -> BN -> ReLU -> MLP2 -> Softmax;
    }
    
    Multiply [label="Element-wise\\nMultiply", shape=circle];
    Normalize [label="L2 Normalize"];
    
    subgraph cluster_text {
        label="Text Features";
        style=filled;
        fillcolor=lightcyan;
        
        Prompts [label="Text Prompts\\n(8 classes)"];
        TextEncoder [label="Text Encoder\\n(冻结)"];
        TextFeatures [label="文本特征\\n[8, 512]"];
        
        Prompts -> TextEncoder -> TextFeatures;
    }
    
    Similarity [label="Cosine Similarity\\nlogit_scale * (img @ txt.T)", shape=ellipse];
    Output [label="输出Logits\\n[B, 8]", shape=parallelogram, fillcolor=lightcoral];
    
    Input -> ImageEncoder;
    ImageFeatures -> MLP1;
    ImageFeatures -> Multiply;
    Softmax -> Multiply;
    Multiply -> Normalize;
    Normalize -> Similarity;
    TextFeatures -> Similarity;
    Similarity -> Output;
}
"""
    
    # 4. GPR-FedSense架构图
    gprfedsense_architecture = """
digraph GPRFedSenseArchitecture {
    rankdir=TB;
    node [shape=box, style="rounded,filled", fillcolor=lightblue, fontname="Arial"];
    
    Input [label="GPR图像\\n224×224×3", shape=parallelogram, fillcolor=lightgreen];
    
    subgraph cluster_local {
        label="Module 1: 本地私有层 (不聚合)";
        style=filled;
        fillcolor=lightyellow;
        
        SignalNorm [label="GPRSignalNorm\\n可学习的 γ,β,gain"];
        Stage1 [label="Stage1\\nConv → BN → ReLU"];
        TimeConv [label="时间域卷积\\n5×1 kernel"];
        SpatialConv [label="空间域卷积\\n1×5 kernel"];
        Fusion [label="特征融合\\nConcat + Conv1×1"];
        
        SignalNorm -> Stage1;
        Stage1 -> TimeConv;
        Stage1 -> SpatialConv;
        TimeConv -> Fusion;
        SpatialConv -> Fusion;
    }
    
    subgraph cluster_shared {
        label="Module 2: 全局共享层 (联邦聚合)";
        style=filled;
        fillcolor=lightcyan;
        
        Backbone [label="Shared Backbone\\nCNN / ResNet18 / MobileViT"];
        AvgPool [label="AdaptiveAvgPool2d\\n→ 512D"];
        
        Backbone -> AvgPool;
    }
    
    subgraph cluster_head {
        label="Module 3: 个性化分类头 (ALA聚合)";
        style=filled;
        fillcolor=lightcoral;
        
        Dropout1 [label="Dropout(0.2)"];
        FC1 [label="FC(512→256)"];
        ReLU1 [label="ReLU"];
        Dropout2 [label="Dropout(0.1)"];
        FC2 [label="FC(256→8)"];
        
        Dropout1 -> FC1 -> ReLU1 -> Dropout2 -> FC2;
    }
    
    Output [label="输出Logits\\n[B, 8]", shape=parallelogram, fillcolor=lightgreen];
    
    Input -> SignalNorm;
    Fusion -> Backbone;
    AvgPool -> Dropout1;
    FC2 -> Output;
    
    // 添加说明
    Note1 [label="设备适配\\n信号归一化", shape=note, fillcolor=white];
    Note2 [label="知识共享\\n通用特征", shape=note, fillcolor=white];
    Note3 [label="个性化\\nNon-IID处理", shape=note, fillcolor=white];
    
    Note1 -> SignalNorm [style=dashed, color=gray];
    Note2 -> Backbone [style=dashed, color=gray];
    Note3 -> FC1 [style=dashed, color=gray];
}
"""
    
    # 保存所有DOT文件
    diagrams = {
        "system_architecture.dot": system_architecture,
        "feddwa_workflow.dot": feddwa_workflow,
        "fedclip_architecture.dot": fedclip_architecture,
        "gprfedsense_architecture.dot": gprfedsense_architecture,
    }
    
    output_dir = "/home/engine/project/architecture_diagrams"
    os.makedirs(output_dir, exist_ok=True)
    
    for filename, content in diagrams.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        print(f"✅ 已创建: {filepath}")
    
    # 生成渲染命令
    print("\n📝 使用以下命令渲染图像 (需要安装graphviz):")
    print("   sudo apt-get install graphviz  # 安装graphviz")
    print(f"\n   cd {output_dir}")
    for filename in diagrams.keys():
        basename = filename.replace('.dot', '')
        print(f"   dot -Tpng {filename} -o {basename}.png")
    
    # 创建批量渲染脚本
    render_script = os.path.join(output_dir, "render_all.sh")
    with open(render_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# 批量渲染架构图\n\n")
        for filename in diagrams.keys():
            basename = filename.replace('.dot', '')
            f.write(f"dot -Tpng {filename} -o {basename}.png\n")
            f.write(f"dot -Tsvg {filename} -o {basename}.svg\n")
        f.write("\necho '✅ 所有图像已生成!'\n")
    
    os.chmod(render_script, 0o755)
    print(f"\n✅ 批量渲染脚本已创建: {render_script}")
    print(f"   运行: bash {render_script}")
    
    return output_dir

if __name__ == "__main__":
    print("=" * 60)
    print("架构图生成工具")
    print("=" * 60)
    
    output_dir = create_graphviz_diagrams()
    
    print("\n" + "=" * 60)
    print("✅ 所有架构图源码已生成!")
    print(f"📁 输出目录: {output_dir}")
    print("=" * 60)
