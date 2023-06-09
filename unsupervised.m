data= readtable('datos/fiducial_normalized_arritmias.txt');
data = removevars(data, 'Var1');
data_matrix = table2array(data);

% Generar un vector de índices aleatorios
n_filas= height(data);
idx = randperm(n_filas);

% Reordenar las filas de la matriz utilizando el vector de índices
data_matrix = data_matrix(idx, :);

%% FUZZY C MEANS
options = fcmOptions(NumClusters=5,Exponent=2,Verbose=false)
[centers,U] = fcm(data_matrix, options);
[~, idx] = max(U, [], 1);

% crear dataset con nuevos labels
data_matrix_con_idx = horzcat(data_matrix, idx');
tabla_con_label = array2table(data_matrix_con_idx, 'VariableNames', [data.Properties.VariableNames, {'label'}]);

% Guardar la tabla como un archivo de texto
nombreArchivo = 'fiducial_normalized_arritmias_labels_fcm_k5.txt';
writetable(tabla_con_label, nombreArchivo, 'Delimiter', '\t');

%% SUBTRACTIVE CLUSTERING
%[centers_sc, sigmas] = subclust(data_matrix, 1);
%ks(i)=size(centers,1)
%D = pdist2(fiducial_matrix, centers_sc, 'euclidean');
%[~, idx_sc] = min(D, [], 2);
%idx_sc=idx_sc'

%%
% Calcular el t-SNE de 3 dimensiones
Y = tsne(data_matrix, 'NumDimensions', 3);
%%
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
%%
% Visualizar los resultados en un gráfico 2D
figure();
scatter(Y(:,1), Y(:,2), [], idx, 'filled');
title('Clustering en t-SNE de 2 dimensiones');
xlabel('Dimensión 1');
ylabel('Dimensión 2');
colormap(jet(5));
colorbar;