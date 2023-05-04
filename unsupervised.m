fiducial= readtable('fiducial_normalized_arritmias.txt');
fiducial = removevars(fiducial, 'Var1');
fiducial_matrix = table2array(fiducial);

% Generar un vector de índices aleatorios
n_filas=18521
idx = randperm(n_filas);

% Reordenar las filas de la matriz utilizando el vector de índices
fiducial_matrix = fiducial_matrix(idx, :);
%%
% Define el rango de valores de c a evaluar
c_range = 2:10;
% Inicializa el vector de valores de índice de silueta
 silhouette_values= zeros(9, 1);
 db_values= zeros(9, 1);
% Realiza Fuzzy C-Means para cada valor de c y calcula el índice de silueta
for i = 1:length(c_range)
    c = c_range(i);
    options = fcmOptions(...
    NumClusters=c,...
    Exponent=2.5,...
    Verbose=false)

    [centers,U] = fcm(fiducial_matrix, options);
    
    [~, idx] = max(U, [], 1);
    silhouette_values(i) = evalclusters(fiducial_matrix, idx', 'silhouette').CriterionValues
    db_values(i) = evalclusters(fiducial_matrix, idx', 'DaviesBouldin').CriterionValues
end

% Visualiza los resultados
figure;
plot(c_range, silhouette_values, '-o');
xlabel('Número de Clusters (c)');
ylabel('Índice de Silueta');
title('Fuzzy C-Means Clustering');

% Visualiza los resultados
figure;
plot(c_range, db_values, '-o');
xlabel('Número de Clusters (c)');
ylabel('Índice de Davies Bouldin');
title('Fuzzy C-Means Clustering');

%% FUZZY C MEANS
options = fcmOptions(NumClusters=3,Exponent=2.5,Verbose=false)
[centers,U] = fcm(fiducial_matrix, options);
[~, idx] = max(U, [], 1);

%% SUBTRACTIVE CLUSTERING
[centers_sc, sigmas] = subclust(fiducial_matrix, 1);
ks(i)=size(centers,1)
D = pdist2(fiducial_matrix, centers_sc, 'euclidean');
[~, idx_sc] = min(D, [], 2);
idx_sc=idx_sc'
%%
figure();
ax = gca();
ax.View = [45, 30];
ax.Box = 'on';
ax.BoxStyle = 'full';
ax.DataAspectRatio = [1, 1, 1];
xlabel('X');
ylabel('Y');
zlabel('Z');

scatter3(fiducial_matrix(:,1), fiducial_matrix(:,2), fiducial_matrix(:,3), [], idx);

hold on;
scatter3(centers(:,1), centers(:,2), centers(:,3), 'k', 'filled');
hold off;

colormap(jet(5));
colorbar;

%%
% Calcular el t-SNE de 3 dimensiones
Y = tsne(fiducial_matrix, 'NumDimensions', 3);

% Visualizar los resultados en un gráfico 3D
figure();
scatter3(Y(:,1), Y(:,2), Y(:,3), [], idx, 'filled');
title('Clustering en t-SNE de 3 dimensiones');
xlabel('Dimensión 1');
ylabel('Dimensión 2');
zlabel('Dimensión 3');
colormap(jet(5));
colorbar;
%%
% Calcular el t-SNE de 2 dimensiones
Y = tsne(fiducial_matrix, 'NumDimensions', 2);

% Visualizar los resultados en un gráfico 2D
figure();
scatter(Y(:,1), Y(:,2), [], idx, 'filled');
title('Clustering en t-SNE de 2 dimensiones');
xlabel('Dimensión 1');
ylabel('Dimensión 2');
colormap(jet(5));
colorbar;