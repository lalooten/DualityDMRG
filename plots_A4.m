%%
% Load dual Hamiltonians
clear;
syms heis lambda mu id;
load('A4model_duals.mat')

N = size(tensors,1);
par = [1 1 1 -10;
       1 -2 -5 -10;
       1 -5 1 -10];
P = size(par,1);

D = 5;
D = D*[3 12 4 12 6 6 6 4 4 4 4 3];
tol = 1e-3;
L = 10;

Energy = cell(N,P);
S = cell(N,P);
A = cell(N,P);
mps_indices = cell(N,P);
truncation_error = cell(N,P);
runtime = zeros(N,P);
Energy_local = cell(N,P);

tensors_run = tensors;
modules = [1 2 3 4 5 8 12];
pars = 1:3;

phases = {'A4_SPT','A4','D2'};

boundary = cell(N,P);
boundary(:) = {[1,1]};
boundary{3,1} = [3,3; 3,3];

% for t = 1:6
%     tol = tol/10;
%     load([mat2str([L tol],4) 'A4.mat']);
for j = pars
    for i = modules
        if ~isempty(A{i,j})
            continue;
        end
        tensors_run{i,2} = double(subs(tensors{i,2},[heis mu lambda id],par(j,:)));
        fprintf('Computing parameter %1i, module %1i\n',j,i);
        tic;
        [Energy{i,j},S{i,j},A{i,j},mps_indices{i,j},truncation_error{i,j},Energy_local{i,j}] = DMRG(tensors_run,i,L,D(i),tol,boundary{i,j},1);
        runtime(i,j) = toc;
        fprintf('Parameter %1i, module %1i took %5f seconds\n',j,i,runtime(i,j));
        fprintf('Energy density: %10.10f, total energy: %10.10f\n',Energy{i,j}(end)/(L-1),Energy{i,j}(end));
    end
    fprintf('\n');
end
% save([mat2str([L tol],4) 'A4_v3.mat'],'Energy','S','A','mps_indices','truncation_error','runtime','Energy_local','par');
% end

%%
% 
mod = {'RepA4','Original','RepPsiA4','RepPsiD2','RepZ2','RepZ2','RepZ2','RepZ3','RepZ3','RepZ3','RepZ3','RepD2'}; %names of module cats
cut = round(L/2);

xmax = zeros(3,1);
for j = pars
    for i = modules
        x = 0;
        for n = 1:length(S{i,j}{cut})
            x = x + sum(diag(S{i,j}{cut}{n})>=1e-3);
        end
        if x > xmax(j)
            xmax(j) = x;
        end
    end
end

for j = pars
    fig = figure('Name',phases{j});
    set(gcf, 'Position',  [100+600*(j-1), 100, 600, 300]);
    tel = 0;
    for i = modules
        tel = tel + 1;
        spec = S{i,j}{cut};
        nb = size(spec,1);
        ss = zeros(nb,1);
        for n = 1:nb
            spec{n} = diag(spec{n});
            % if and(i==1,n==4)
            %     spec{n} = spec{n}/sqrt(3);
            % end
            ss(n) = size(spec{n},1);
        end
        spec = cell2mat(spec);
        spec = spec/sqrt(sum(spec.^2));
        [~,ind] = sort(spec,'descend');
        [~,ind] = sort(ind);
        subplot(4,2,tel);
        s = [0; cumsum(ss)];
        hold on;
        entropy = 0;
        for bl = 1:nb
            if ss(bl) > 0
                plot(ind(1+s(bl):s(bl)+ss(bl)),log10(spec(1+s(bl):s(bl)+ss(bl))),'.');
                entropy = entropy - sum(spec(1+s(bl):s(bl)+ss(bl)).*log(spec(1+s(bl):s(bl)+ss(bl))));
            end
        end
        title([mod{i} ', S = ' num2str(entropy)]);
        ylim([-3 0]);
        xlim([0 xmax(j)]);
        hold off;
    end
end