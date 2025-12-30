clear;
clc;
warning off;
addpath(genpath('./'));

%% dataset
ds = './data/Wiki_fea.mat';
metric = {'ACC','nmi','Purity','Fscore','Precision','Recall','AR','Entropy'};


    %% load data & make folder

    load(ds);
   %X = fea;
   %Y = gnd;
    k = length(unique(Y));
    n = length(Y);


    for iv = 1:length(X)
        if size(X{iv},2)~= n
            X{iv} = X{iv}';
        end
        X{iv} = NormalizeFea(X{iv}, 0);
    end
    %% param setting
    c =6;%[1,2,3,4,5, 6,7,8,10,12];
    AP = 1e-1;%[ 1e-1, 1e0, 1e1, 1e2,1e3,1e4];
    BT =1e2;%[1e1,1e2, 1e3];
    CT = 1e1;%[1e1,1e2, 1e3];
    %% 
 
         param.lambda1 = AP;
         param.lambda2 = BT;
         param.lambda3 = CT;
         tic
         [P, A, Z,S, M, E] = TLF_MVC(X, n, k, c, param);
         [res] = myNMIACCwithmean(P,Y,k); %[ACC nmi Purity Fscore Precision Recall AR Entropy];
         time  = toc;   
         disp(time);
         fprintf("\n ACC, NMI, Purity, F, AR: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f \n",  res(1), res(2),res(3),res(4),res(7));
      
  

%% 1. 数据准备
% 确保这里的 labels 变量是包含 5 个类别的原始标签 (n x 1)
%data = P; 
% 注意：如果你的标签变量原本叫 Y，请改个名字，比如叫 true_labels
%true_labels = Y; 

%% 2. 运行 t-SNE
%fprintf('正在计算 t-SNE...\n');
% 将结果存为 Y_tsne，不要直接存为 Y，避免覆盖 labels
%Y_tsne = tsne(data, 'Algorithm', 'exact', 'Standardize', true, 'Perplexity', 5);

%% 3. 绘图
%fig = figure('Color', 'w');
%set(fig, 'Position', [60, 60, 800, 600]);  % 800x600像素
% 使用 Y_tsne 作为坐标，true_labels 作为上色依据
%gscatter(Y_tsne(:,1), Y_tsne(:,2), true_labels); 

%% 4. 美化图片
%set(gca, 'FontSize', 5, 'FontName', 'Times New Roman');
%legend off;
%axis on; 
%grid off;
%legend('Location', 'northeastoutside');

%% 5. 导出
%exportgraphics(gca, 'Wiki-clustering.png', 'Resolution', 500);





















%X{1} = X{2};
%A{1} = A{2};

% 假设 X_v 是原始视图数据 [d x n]，A_v 是学习到的锚点 [d x m]
% 1. 拼接
%combined_data = [X{1}, A{1}]'; % 得到 (n+m) x d

% 2. 创建一个区分标志
% 前 n 个点标记为 'Sample'，后 m 个点标记为 'Anchor'
%group_idx = [repmat({'Sample'}, size(X{1},2), 1); repmat({'Anchor'}, size(A{1},2), 1)];

% 3. 跑 t-SNE
%Y_mixed = tsne(combined_data, 'Perplexity', 30);

% 4. 绘图
%figure;
%hold on;
% 画样本（灰色，小点）
%scatter(Y_mixed(1:size(X{1},2), 1), Y_mixed(1:size(X{1},2), 2), 10,Y, 'filled', 'MarkerFaceAlpha', 0.8);
% 画锚点（彩色，大点，带边框）
% 将 'p' 添加到参数中，并建议将大小 20 改为 80 以增强视觉效果
%scatter(Y_mixed(size(X{1},2)+1:end, 1), Y_mixed(size(X{1},2)+1:end, 2), 60, 'r', 'p', 'filled', ...
   %     'MarkerEdgeColor', 'none');
%legend('Raw Samples', 'Learned Anchors');
%hold off;
%exportgraphics(gca, 'MSRC-cluster.png', 'Resolution', 500);