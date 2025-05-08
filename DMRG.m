function [Energy,S,A,mps_indices,truncation_error,Energy_local] = DMRG2(tensors,modul,L,chi,toler,boundary_mod,disp,A)

% Extract tensors for the given module
w1 = tensors{modul, 1}; 
w2 = tensors{modul, 2}; 
w3 = tensors{modul, 3}; 

% Number of blocks
nblock = w3(1); 
J = 1:w3(1); 

% Filter out small entries in w2 and adjust w1 and w2 accordingly
[f, ~] = find(abs(w2) > 1e-13); 
w1 = w1(f, :); 
w2 = w2(f, :); 
T = w2;

% Ensure boundary_mod is a row vector
if size(boundary_mod, 1) > size(boundary_mod, 2)
    boundary_mod = boundary_mod.';
end

% Adjust chi to match the size of w3(1)
if numel(chi) ~= w3(1)
    chi = chi * ones(1, w3(1));
end
chimax = chi; 
chi = 3 * ones(size(chi)) + (-1:numel(chi) - 2);

% Handle cases where chi contains zeros
if sum(chi == 0) > 0
    [~, ff] = find(chi == 0); 
    chi(ff) = []; 
    fff = zeros(size(w1, 1), 1);
    
    % Filter rows in w1 where specific indices match ff
    for k = 1:numel(ff)
        for kk = [1, 3, 5, 7, 9, 11, 13, 15]
            fff = fff + (w1(:, kk) == ff(k));
        end
    end
    [f, ~] = find(fff == 0); 
    w1 = w1(f, :); 
    w2 = w2(f, :); 
    T = w2; 
    J = 1:sum(chi > 0);
end

% Normalize w1 values for each column
for kk = 1:16
    [~, ~, q] = unique(w1(:, kk)); 
    w1(:, kk) = q;
end

% Initialize chi_cell
chi_cell = cell(L, 2);
for k = 1:L
    chi_cell(k, 1) = {chi};
    chi_cell(k, 2) = {chi};
end
chi_cell(1, 1) = {ones(1, nblock)};
chi_cell(L, 2) = {ones(1, nblock)};

% Rearrange columns of w1
w1 = w1(:, [13, 14, 16, 15, 9, 10, 12, 11, 5, 6, 8, 7, 1, 2, 4, 3]);

% Create unique indices for w1 combinations
ww = w1(:, 1:4); 
wm = max(vec(ww)) + 7; 
www = ww * wm.^((0:3)'); 
[f, ff] = unique(www);

% Map indices to tensor blocks
ww1 = w1 * kron(eye(4), wm.^((0:3)')); 
Tind = zeros(size(ww1));
for k = 1:numel(f)
    Tind = Tind + (ww1 == f(k)) * k;
end

% Prepare MPS indices and tensor indices
mps_indices = cell(numel(f), 1);
for k = 1:numel(f)
    mps_indices(k, 1) = {ww(ff(k), :)};
end

indT = cell(size(Tind, 1), 1);
for k = 1:size(Tind, 1)
    indT(k, 1) = {Tind(k, :)};
end

% Initialize connection arrays
Jl = cell(max(J), 1); 
Jr = Jl; 
Jlmps = Jl; 
Jrmps = Jr;

% Populate connections for MPS
for k = 1:numel(mps_indices)
    m1 = mps_indices{k}(1); 
    m4 = mps_indices{k}(4);

    % Update left and right connections
    Jl(m1) = {unique([Jl{m1}, m4])}; 
    Jlmps(m1) = {unique([Jlmps{m1}, k])};
    Jr(m4) = {unique([Jr{m4}, m1])}; 
    Jrmps(m4) = {unique([Jrmps{m4}, k])};
end

% Number of MPS indices and initialize variables
nA = numel(mps_indices);
mps_ind = mps_indices;
mps2_i = {};
tel = 0;

% Build the mps2_i array and initialize lookup table
for k = 1:numel(mps_ind)
    q = mps_ind{k};
    for kk = Jlmps{q(4)}
        tel = tel + 1;
        mps2_i(tel, 1) = {[k, kk]};
    end
end
NN = tel;
lookup_mps2_i = zeros(NN, NN); % Initialize lookup table
for k = 1:tel
    lookup_mps2_i(mps2_i{k}(1), mps2_i{k}(2)) = k;
end

% Create ind_middle_is_j based on mps2_i
ind_middle_is_j = cell(numel(J), 1);
for k = 1:numel(mps2_i)
    ind_middle_is_j{mps_ind{mps2_i{k}(1)}(4)} = ...
        [ind_middle_is_j{mps_ind{mps2_i{k}(1)}(4)}, k];
end

% Create Jr_mps2 and Jl_mps2 mappings
Jr_mps2 = cell(nA, 1);
Jl_mps2 = cell(nA, 1);
for k = 1:numel(mps2_i)
    Jr_mps2{mps2_i{k}(2)} = [Jr_mps2{mps2_i{k}(2)}, k];
    Jl_mps2{mps2_i{k}(1)} = [Jl_mps2{mps2_i{k}(1)}, k];
end

% Build indT2R and T2R arrays
indT2R = [];
T2R = [];
tel = 0;
for k = 1:numel(indT)
    q = indT{k};
    [~, ffu] = find(lookup_mps2_i(q(3), :) ~= 0);
    [~, ffd] = find(lookup_mps2_i(q(2), :) ~= 0);
    fff = intersect(ffu, ffd);
    for kk = 1:numel(fff)
        tel = tel + 1;
        indT2R(tel, :) = [q(1), lookup_mps2_i(q(2), fff(kk)), lookup_mps2_i(q(3), fff(kk)), q(4)];
        T2R(tel, 1) = T(k);
    end
end

% Build indT2L array
indT2L = [];
tel = 0;
for k = 1:numel(indT)
    q = indT{k};
    [ffu, ~] = find(lookup_mps2_i(:, q(1)) ~= 0);
    [ffd, ~] = find(lookup_mps2_i(:, q(4)) ~= 0);
    fff = intersect(ffu, ffd);
    for kk = 1:numel(fff)
        tel = tel + 1;
        indT2L(tel, :) = [lookup_mps2_i(fff(kk), q(1)), q(2), q(3), lookup_mps2_i(fff(kk), q(4))];
        T2L(tel, 1) = T(k);
    end
end

% Create mappings for indT2R_x and indT2L_x
indT2R_x = cell(numel(mps2_i), 1);
for k = 1:numel(mps2_i)
    [ff, ~] = find(indT2R(:, 2) == k);
    indT2R_x{k} = ff;
end

indT2L_x = cell(numel(mps2_i), 1);
for k = 1:numel(mps2_i)
    [ff, ~] = find(indT2L(:, 4) == k);
    indT2L_x{k} = ff;
end

% Build indH2 mappings
indH2 = cell(numel(mps2_i), 1);
for k = 1:numel(mps2_i)
    [ff, ~] = find((Tind(:, 4) == mps2_i{k}(1)) .* (Tind(:, 2) == mps2_i{k}(2)));
    indH2{k, 1} = ff;
end

% Generate SVD-related indices
[indxAA, NindxAA, ~, ~, ~] = svd_all_indices(chi, chi, mps_ind, mps2_i, nblock, ind_middle_is_j, NN);

% Initialize a zero cell array for NN x NN
cellzeroNN = cell(NN, NN);
for k = 1:NN
    for kk = 1:NN
        cellzeroNN{k, kk} = 0;
    end
end

% Build telTRr and telTLl mappings
telTRr = cell(NN, 1);
for kk = 1:size(indT2R, 1)
    q = indT2R(kk, :);
    telTRr{q(2)} = unique([telTRr{q(2)}, q(3)]);
end

telTLl = cell(NN, 1);
for kk = 1:size(indT2L, 1)
    q = indT2L(kk, :);
    telTLl{q(4)} = unique([telTLl{q(4)}, q(1)]);
end

indices_AA = {mps2_i, indT2R, indT2L, indT2R_x, indT2L_x, indH2, indxAA, NindxAA, lookup_mps2_i, Jl_mps2, Jr_mps2, telTRr, telTLl};

% Initialize A if not already defined
if ~exist('A', 'var')
    % Define random A matrices with proper normalization
    A = cell(L, 1);
    for n = 1:L
        AA = cell(nA, 1);
        for k = 1:nA
            row_size = chi_cell{n, 1}(mps_ind{k}(1));
            col_size = chi_cell{n, 2}(mps_ind{k}(4));
            normalization_factor = sqrt(row_size) + sqrt(col_size);
            AA{k} = (1 + randn(row_size, col_size)) / normalization_factor;
        end
        A{n} = AA;
    end

    % Apply boundary conditions if nblock > 1
    if nblock > 1
        % Modify the left boundary
        Y = A{1};
        for k = 1:nA
            if mps_indices{k}(1) ~= boundary_mod(1, 1)
                Y{k} = zeros(size(Y{k}));
            end
        end
        A{1} = Y;

        % Modify the right boundary
        Y = A{L};
        for k = 1:nA
            if mps_indices{k}(4) ~= boundary_mod(1, 2)
                Y{k} = zeros(size(Y{k}));
            end
        end
        A{L} = Y;
    end

else
    % Ensure A matrices have valid dimensions and update chi_cell
    for n = 1:L
        chiil = zeros(1, nA);
        chiir = zeros(1, nA);
        for k = 1:nA
            chiil(k) = size(A{n}{k}, 1);
            chiir(k) = size(A{n}{k}, 2);

            % Replace invalid dimensions with zeroed matrices
            if chiil(k) == 0 || chiir(k) == 0
                A{n}{k} = zeros(max(1, chiil(k)), max(1, chiir(k)));
            end

            % Update dimensions to be at least 1
            chiil(k) = max(chiil(k), 1);
            chiir(k) = max(chiir(k), 1);
        end
        chi_cell{n, 1} = chiil;
        chi_cell{n, 2} = chiir;
    end
end

% Initialize AA0 with zero matrices for all MPS components
AA0 = cell(NN, 1);
for k = 1:NN
    ql = mps_ind{mps2_i{k}(1)}(1);
    qr = mps_ind{mps2_i{k}(2)}(4);
    AA0{k} = zeros(chi(ql), chi(qr));
end

% Bring MPS into right canonical form
for n = L:-1:2
    % Perform QR decomposition
    [Q, R] = QR_mps_right(A{n}, Jlmps);
    A{n} = Q;
    
    % Update the previous site with the contraction of R
    Q_prev = cell(nA, 1);
    for k = 1:nA
        q = mps_ind{k};
        Q_prev{k} = A{n-1}{k} * R{q(4)};
    end
    A{n-1} = Q_prev;
end

% Normalize A{1}
norm_factor = 0;
for k = 1:nA
    norm_factor = norm_factor + norm(A{1}{k}, "fro")^2;
end
norm_factor = sqrt(norm_factor);

Q_normalized = cell(nA, 1);
for k = 1:nA
    Q_normalized{k} = A{1}{k} / norm_factor;
end
A{1} = Q_normalized;

% Initialize reduced density matrices (rho_left and rho_right)
rhol = cell(L, 1);
rhor = cell(L, 1);
Q0 = cell(nblock, 1);
for k = 1:nblock
    Q0{k} = 0; % Initialize with zeros
end
for n = 1:L
    rhol{n} = Q0;
    rhor{n} = Q0;
end

% Initialize rho_0 with identity matrices
rho_0 = cell(nblock, 1);
for k = 1:nblock
    rho_0{k} = 1;
end

% Compute rhol (left reduced density matrices)
Q = Q0;
for k = 1:nA
    q = mps_ind{k};
    Q{q(4)} = Q{q(4)} + A{1}{k}' * rho_0{q(1)} * A{1}{k};
end
rhol{1} = Q;

for n = 2:L-1
    Q = Q0;
    for k = 1:nA
        q = mps_ind{k};
        Q{q(4)} = Q{q(4)} + A{n}{k}' * rhol{n-1}{q(1)} * A{n}{k};
    end
    rhol{n} = Q;
end

% Compute rhor (right reduced density matrices)
Q = Q0;
for k = 1:nA
    q = mps_ind{k};
    Q{q(1)} = Q{q(1)} + A{L}{k} * rho_0{q(4)} * A{L}{k}';
end
rhor{L} = Q;

for n = L-1:-1:2
    Q = Q0;
    for k = 1:nA
        q = mps_ind{k};
        Q{q(1)} = Q{q(1)} + A{n}{k} * rhor{n+1}{q(4)} * A{n}{k}';
    end
    rhor{n} = Q;
end

% Compute El and Er (effective environments)
El = cell(L, 1);
Er = cell(L, 1);

for n = 2:L-1
    El{n} = ElEr_module_mps('left', n, L, A, rhol, rhor, rho_0, El, Er, ...
                             mps_ind, indT, T, Jrmps, Jlmps, nblock, Q0);
end

for n = L-1:-1:2
    Er{n} = ElEr_module_mps('right', n, L, A, rhol, rhor, rho_0, El, Er, ...
                             mps_ind, indT, T, Jrmps, Jlmps, nblock, Q0);
end

% Initialize S matrices
S = cell(L-1, 1);
S0 = cell(nblock, 1);
for k = 1:nblock
    S0{k} = diag([1, zeros(1, chi(k)-1)]);
end

for n = 1:L-1
    S{n} = cell(nblock, 1);
    for k = 1:nblock
        S{n}{k} = 1; % Simplified initialization
    end
end

% Optional testing (commented out)
% test1: f = []; for k = 1:L-1
%     ff = 0;
%     for kk = 1:nblock
%         ff = ff + trace(rhol{k}{kk} * rhor{k+1}{kk});
%     end
%     f(k) = ff;
% end
% f - mean(f)

% test2: f = []; for k = 2:L-1
%     ff = 0;
%     for kk = 1:nblock
%         ff = ff + trace(El{k}{kk} * rhor{k+1}{kk}) + trace(rhol{k-1}{kk} * Er{k}{kk});
%     end
%     f(k-1) = ff;
% end
% f - mean(f)

% End of MPS initialization

% Options for eigenvalue solver
opts.tol = 1e-14; % Tolerance for eigenvalue solver
opts.issym = 1;   % Ensure the problem is symmetric
opts.disp = 0;    % Suppress solver output

% Loop for DMRG: 3 complete sweeps back and forth
Energy = [];
tell = 0;
truncation_error = zeros(L-1, 1);
Energy_local = zeros(L-1, 1);
nold = 0;

% Sweep pattern: Full sweep followed by central sweeps
sweep_pattern = [1:L-1, L-2:-1:1, 2:L-1, L-2:-1:1, 2:L-1, L-2:-1:1, 2:L-1, L-2:-1:1];
for n = sweep_pattern
    tell = tell + 1;

    % Compute TRr
    TRr = cellzeroNN;
    if n > 2
        for kk = 1:size(indT2R, 1)
            q = indT2R(kk, :);
            qll = mps_ind{q(1)};
            TRr(q(2), q(3)) = {TRr{q(2), q(3)} + T2R(kk) * A{n-1}{q(4)}' * rhol{n-2}{qll(1)} * A{n-1}{q(1)}};
        end
    elseif n == 2
        for kk = 1:size(indT2R, 1)
            q = indT2R(kk, :);
            qll = mps_ind{q(1)};
            TRr(q(2), q(3)) = {TRr{q(2), q(3)} + T2R(kk) * A{n-1}{q(4)}' * rho_0{qll(1)} * A{n-1}{q(1)}};
        end
    end

    % Compute TLl
    TLl = cellzeroNN;
    if n < L-2
        for kk = 1:size(indT2L, 1)
            q = indT2L(kk, :);
            qrr = mps_ind{q(2)};
            TLl(q(4), q(1)) = {TLl{q(4), q(1)} + T2L(kk) * A{n+2}{q(3)} * rhor{n+3}{qrr(4)} * A{n+2}{q(2)}'};
        end
    elseif n == L-2
        for kk = 1:size(indT2L, 1)
            q = indT2L(kk, :);
            qrr = mps_ind{q(2)};
            TLl(q(4), q(1)) = {TLl{q(4), q(1)} + T2L(kk) * A{n+2}{q(3)} * rho_0{qrr(4)} * A{n+2}{q(2)}'};
        end
    end

    % Extract relevant blocks
    if n > 1
        Eln1 = El{n-1};
        rholn1 = rhol{n-1};
    else
        Eln1 = [];
        rholn1 = rho_0;
    end

    if n < L-1
        Ern2 = Er{n+2};
        rhorn2 = rhor{n+2};
    else
        Ern2 = [];
        rhorn2 = rho_0;
    end

    % Compute SVD indices
    chi_l = chi_cell{n, 1};
    chi_r = chi_cell{n+1, 2};
    [indxAA, NindxAA, svd_ind, svd_ind_l, svd_ind_r] = svd_all_indices(chi_l, chi_r, mps_ind, mps2_i, nblock, ind_middle_is_j, NN);

    % Prepare initial guess for eigensolver
    if tell > L+1 && n ~= 1 && n ~= (L-1)
        if n > nold
            for k = 1:NN
                q = mps2_i{k};
                AAA(k) = {A{n}{q(1)} * A{n+1}{q(2)} * S{n+1}{mps_ind{q(2)}(4)}};
            end
        else
            for k = 1:NN
                q = mps2_i{k};
                AAA(k) = {S{n-1}{mps_ind{q(1)}(1)} * A{n}{q(1)} * A{n+1}{q(2)}};
            end
        end
        x0 = [];
        for telt = 1:NN
            qq = mps2_i{telt};
            ql = mps_ind{qq(1)};
            qr = mps_ind{qq(2)};
            x0(indxAA{telt}) = reshape(AAA{telt}, [chi_l(ql(1)) * chi_r(qr(4)), 1]);
        end
        opts.v0 = x0.';
    else
        opts.v0 = randn(NindxAA, 1);
    end

    % Solve for ground state
    [x, ~] = eigs(@(x)module_dmrg_AA(x, n, L, AA0, Eln1, Ern2, rholn1, rhorn2, TRr, TLl, indices_AA, T, mps_ind, indT, chi_l, chi_r, indxAA, boundary_mod, tell), NindxAA, 1, 'lr', opts);
    x = x(:, 1);

    % Update tensors
    AA = cell(NN, 1);
    for k = 1:NN
        ql = mps_ind{mps2_i{k}(1)}(1);
        qr = mps_ind{mps2_i{k}(2)}(4);
        AA(k) = {reshape(x(indxAA{k}, 1), [chi_l(ql), chi_r(qr)])};
    end

    % Perform SVD decomposition
    [Al, ss, Ar] = svd_module_mps2(AA, nblock, ind_middle_is_j, svd_ind, chi_cell{n, 2}, mps2_i, svd_ind_l, svd_ind_r, chimax, toler);
    A(n) = {Al};
    A(n+1) = {Ar};
    S(n) = {ss};

    % Update chi values
    for k = 1:nblock
        chi_cell{n, 2}(k) = size(ss{k}, 1);
        chi_cell{n+1, 1}(k) = size(ss{k}, 1);
    end

    % Calculate truncation error
    tr = 0;
    AAA = AA;
    for k = 1:numel(AA)
        q = mps2_i{k};
        AAA(k) = {Al{q(1)} * ss{mps_ind{q(1)}(4)} * Ar{q(2)}};
        tr = tr + norm(AAA{k} - AA{k});
    end
    truncation_error(n) = tr;

    % Calculate energy
    xt = [];
    for telt = 1:NN
        qq = mps2_i{telt};
        ql = mps_ind{qq(1)};
        qr = mps_ind{qq(2)};
        xt(indxAA{telt}) = reshape(AAA{telt}, [chi_l(ql(1)) * chi_r(qr(4)), 1]);
    end
    Energy(tell) = module_dmrg_AA(xt, n, L, AA0, Eln1, Ern2, rholn1, rhorn2, TRr, TLl, indices_AA, T, mps_ind, indT, chi_l, chi_r, indxAA, boundary_mod, tell) * xt';
    Energy_local(n) = module_dmrg_AA_local(xt, n, L, AA0, Eln1, Ern2, rholn1, rhorn2, TRr, TLl, indices_AA, T, mps_ind, indT, chi_l, chi_r, indxAA, boundary_mod) * xt';

    % Update reduced density matrices
    if n == 1
        Q = Q0;
        for k = 1:nA
            q = mps_ind{k};
            Q(q(4)) = {Q{q(4)} + A{1}{k}' * rho_0{q(1)} * A{1}{k}};
        end
        rhol(1) = {Q};
    elseif n > 1
        Q = Q0;
        for k = 1:nA
            q = mps_ind{k};
            Q(q(4)) = {Q{q(4)} + A{n}{k}' * rhol{n-1}{q(1)} * A{n}{k}};
        end
        rhol(n) = {Q};
    end

    if n == L-1
        Q = Q0;
        for k = 1:nA
            q = mps_ind{k};
            Q(q(1)) = {Q{q(1)} + A{L}{k} * rho_0{q(4)} * A{L}{k}'};
        end
        rhor(L) = {Q};
    elseif n < L-1
        Q = Q0;
        for k = 1:nA
            q = mps_ind{k};
            Q(q(1)) = {Q{q(1)} + A{n+1}{k} * rhor{n+2}{q(4)} * A{n+1}{k}'};
        end
        rhor(n+1) = {Q};
    end

    % Update environment tensors
    El(n) = {ElEr_module_mps('left', n, L, A, rhol, rhor, rho_0, El, Er, mps_ind, indT, T, Jrmps, Jlmps, nblock, Q0)};
    Er(n+1) = {ElEr_module_mps('right', n+1, L, A, rhol, rhor, rho_0, El, Er, mps_ind, indT, T, Jrmps, Jlmps, nblock, Q0)};

    % Display energy if required
    if disp ~= 0 && n == 1
        display([n, real(Energy(tell)) / (L-1), tr]);
    end

    nold = n;
end

% Final adjustments to tensors
Q = Q0;
for k = 1:nA
    q = mps_indices{k};
    Q(k, 1) = {A{n}{k} * S{n}{q(4)}};
end
A(n) = {Q};


%%%%%%%%%%%EXTRA FUNCTIONS

% Main function: Effective Hamiltonian on 2 sites
function x = module_dmrg_AA(x, n, L, AA, Eln1, Ern2, rholn1, rhorn2, TRr, TLl, indices_AA, T, mps_ind, indT, chi_l, chi_r, indxAA, boundary_mod, tell)

    % Extract indices and helper variables
    mps2_i = indices_AA{1};
    indH2 = indices_AA{6};
    lookup_mps2_i = indices_AA{9};
    telTRr = indices_AA{12};
    telTLl = indices_AA{13};
    NN = numel(indxAA);

    % Reshape input vector x into tensor AA
    tel = 0;
    for k = 1:NN
        ql = mps_ind{mps2_i{k}(1)}(1);
        qr = mps_ind{mps2_i{k}(2)}(4);
        AA{k} = reshape(x(tel + (1:chi_l(ql) * chi_r(qr))), [chi_l(ql), chi_r(qr)]);
        tel = tel + chi_l(ql) * chi_r(qr);
    end

    % Compute intermediate tensors AAr, AAl, and AAlr
    AAr = cell(NN, 1);
    AAl = cell(NN, 1);
    AAlr = cell(NN, 1);

    for kk = 1:NN
        ql = mps_ind{mps2_i{kk}(1)}(1);
        qr = mps_ind{mps2_i{kk}(2)}(4);
        AAr{kk} = AA{kk} * rhorn2{qr};
        AAl{kk} = rholn1{ql} * AA{kk};
        AAlr{kk} = AAl{kk} * rhorn2{qr};
    end

    % Initialize output vector x
    for tel = 1:NN
        qq = mps2_i{tel};
        ql = mps_ind{qq(1)};
        qr = mps_ind{qq(2)};
        QQ = zeros(chi_l(ql(1)), chi_r(qr(4))); % Initialize local block

        % Apply boundary conditions
        if n == 1 && ql(1) ~= boundary_mod(1, 1)
            QQ = QQ - 1e5 * AAlr{tel};
        elseif n == L-1 && qr(4) ~= boundary_mod(1, 2)
            QQ = QQ - 1e5 * AAlr{tel};
        end

        if tell < L+2 && size(boundary_mod, 1) == 2
            if n == 2 && ql(1) ~= boundary_mod(2, 1)
                QQ = QQ - 1e5 * AAlr{tel};
            elseif n == L-2 && qr(4) ~= boundary_mod(2, 2)
                QQ = QQ - 1e5 * AAlr{tel};
            end
        end

        % Add contributions from Hamiltonian terms
        if n > 2
            QQ = QQ + Eln1{ql(1)} * AAr{tel};
        end
        if n < L-2
            QQ = QQ + AAl{tel} * Ern2{qr(4)};
        end

        % Add contributions from local two-site Hamiltonian
        for k = 1:numel(indH2{tel})
            qq = indH2{tel}(k);
            qqq = indT{qq};
            QQ = QQ + T(qq) * AAlr{lookup_mps2_i(qqq(1), qqq(3))};
        end

        % Add contributions from transfer matrices
        if n > 1
            for k = 1:numel(telTRr{tel})
                kk = telTRr{tel}(k);
                QQ = QQ + TRr{tel, kk} * AAr{kk};
            end
        end
        if n < L-1
            for k = 1:numel(telTLl{tel})
                kk = telTLl{tel}(k);
                QQ = QQ + AAl{kk} * TLl{tel, kk};
            end
        end

        % Store the reshaped result in the output vector
        x(indxAA{tel}) = reshape(QQ, [chi_l(ql(1)) * chi_r(qr(4)), 1]);
    end



function x = module_dmrg_AA_local(x, ~, ~, ~, ~, ~, rholn1, rhorn2, ~, ~, indices_AA, T, mps_ind, indT, chi_l, chi_r, indxAA, ~)

    % Extract indices and helper variables
    mps2_i = indices_AA{1};
    indH2 = indices_AA{6};
    lookup_mps2_i = indices_AA{9};
    NN = numel(indxAA);

    % Reshape input vector x into tensor AA
    tel = 0;
    for k = 1:NN
        ql = mps_ind{mps2_i{k}(1)}(1);
        qr = mps_ind{mps2_i{k}(2)}(4);
        AA{k} = reshape(x(tel + (1:chi_l(ql) * chi_r(qr))), [chi_l(ql), chi_r(qr)]);
        tel = tel + chi_l(ql) * chi_r(qr);
    end

    % Compute intermediate tensors AAr, AAl, and AAlr
    AAr = cell(NN, 1);
    AAl = cell(NN, 1);
    AAlr = cell(NN, 1);

    for kk = 1:NN
        ql = mps_ind{mps2_i{kk}(1)}(1);
        qr = mps_ind{mps2_i{kk}(2)}(4);
        AAr{kk} = AA{kk} * rhorn2{qr};
        AAl{kk} = rholn1{ql} * AA{kk};
        AAlr{kk} = AAl{kk} * rhorn2{qr};
    end

    % Update output vector x using local operations
    for tel = 1:NN
        qq = mps2_i{tel};
        ql = mps_ind{qq(1)};
        qr = mps_ind{qq(2)};
        QQ = zeros(chi_l(ql(1)), chi_r(qr(4))); % Initialize local block

        % Add contributions from Hamiltonian terms
        for k = 1:numel(indH2{tel})
            qq = indH2{tel}(k);
            qqq = indT{qq};
            QQ = QQ + T(qq) * AAlr{lookup_mps2_i(qqq(1), qqq(3))};
        end

        % Store the reshaped result in the output vector
        x(indxAA{tel}) = reshape(QQ, [chi_l(ql(1)) * chi_r(qr(4)), 1]);
    end


function [Al, ss, Ar] = svd_module_mps2(AA, nblock, ind_middle_is_j, svd_ind, ~, mps2_i, svd_ind_l, svd_ind_r, chimax, toler)

    % Initialize outputs
    Al = {};
    Ar = {};
    ss = {};

    % Loop over each block to perform SVD
    for k = 1:nblock
        % Assemble the block matrix AAA
        AAA = [];
        q = ind_middle_is_j{k};
        for kk = 1:numel(q)
            AAA(svd_ind{k}{kk, 1}, svd_ind{k}{kk, 2}) = AA{q(kk)};
        end

        % Perform SVD on AAA
        [u, s, v] = svd(AAA, "econ");
        schi = sum(diag(s) > toler); % Determine the number of significant singular values
        schi = min(schi, chimax(k)); % Enforce maximum bond dimension limit

        % Truncate SVD results to `schi`
        ss{k, 1} = s(1:schi, 1:schi); % Store truncated singular values
        u = u(:, 1:schi);
        v = v(:, 1:schi);

        % Assign left and right singular vectors to Al and Ar
        for kk = 1:numel(q)
            qqq = mps2_i{q(kk)}(1);
            Al(qqq, 1) = {u(svd_ind_l{k}{qqq}, :)};
        end
        for kk = 1:numel(q)
            qqq = mps2_i{q(kk)}(2);
            Ar(qqq, 1) = {v(svd_ind_r{k}{qqq}, :)'};
        end
    end

    % Normalize singular values across all blocks
    total_norm = 0;
    for k = 1:nblock
        total_norm = total_norm + sum(diag(ss{k}).^2);
    end
    total_norm = sqrt(total_norm);
    for k = 1:nblock
        ss{k} = ss{k} / total_norm;
    end

    % (Optional test code for debugging)
    % tr = 0;
    % for k = 1:numel(AA)
    %     q = mps2_i{k};
    %     tr = tr + norm(Al{q(1)} * ss{mps_ind{q(1)}(4)} * Ar{q(2)} - AA{k});
    % end
    % tr


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function Q = ElEr_module_mps(leftright, n, L, A, rhol, rhor, rho_0, El, Er, mps_ind, indT, T, Jrmps, Jlmps, nblock, Q0)
    % Initialize Q from Q0
    Q = Q0;

    % Process for 'left'
    if numel(leftright) == 4  % 'left'
        if n == 2
            % Special case for n = 2
            for tel = 1:numel(indT)
                q = indT{tel};
                qr = mps_ind{q(2)}(4);
                ql = mps_ind{q(1)}(1);
                Q{qr} = Q{qr} + T(tel) * A{2}{q(2)}' * A{1}{q(4)}' * rho_0{ql} * A{1}{q(1)} * A{2}{q(3)};
            end
        elseif n > 2
            % General case for n > 2
            for tel = 1:numel(indT)
                q = indT{tel};
                qr = mps_ind{q(2)}(4);
                ql = mps_ind{q(1)}(1);
                Q{qr} = Q{qr} + T(tel) * A{n}{q(2)}' * A{n-1}{q(4)}' * rhol{n-2}{ql} * A{n-1}{q(1)} * A{n}{q(3)};
            end

            % Add contributions from El
            for qr = 1:nblock
                for k = Jrmps{qr}
                    q = mps_ind{k};
                    Q{qr} = Q{qr} + A{n}{k}' * El{n-1}{q(1)} * A{n}{k};
                end
            end
        end
    end

    % Process for 'right'
    if numel(leftright) == 5  % 'right'
        if n == L - 1
            % Special case for n = L - 1
            for tel = 1:numel(indT)
                q = indT{tel};
                ql = mps_ind{q(1)}(1);
                qr = mps_ind{q(3)}(4);
                Q{ql} = Q{ql} + T(tel) * A{L-1}{q(1)} * A{L}{q(3)} * rho_0{qr} * A{L}{q(2)}' * A{L-1}{q(4)}';
            end
        elseif n < L - 1
            % General case for n < L - 1
            for tel = 1:numel(indT)
                q = indT{tel};
                ql = mps_ind{q(1)}(1);
                qr = mps_ind{q(3)}(4);
                Q{ql} = Q{ql} + T(tel) * A{n}{q(1)} * A{n+1}{q(3)} * rhor{n+2}{qr} * A{n+1}{q(2)}' * A{n}{q(4)}';
            end

            % Add contributions from Er
            for ql = 1:nblock
                for k = Jlmps{ql}
                    q = mps_ind{k};
                    Q{ql} = Q{ql} + A{n}{k} * Er{n+1}{q(4)} * A{n}{k}';
                end
            end
        end
    end


%%%%%% QR to the right
function [A, R] = QR_mps_right(A, Jlmps)
    % Initialize Q and R
    Q = cell(numel(Jlmps), 1);
    R = cell(numel(Jlmps), 1);

    for k = 1:numel(Jlmps)
        z = Jlmps{k};
        
        % Concatenate tensors along rows
        for kk = z
            Q{k} = [Q{k}, A{kk}];
        end
        
        % Perform QR decomposition (via SVD-like function)
        [u, r] = polsvd_c(Q{k}.');
        u = u.'; % Transpose back for consistency
        r = r.'; % Transpose R for consistency
        
        % Distribute components back to A
        tel = 0;
        for kk = z
            A{kk} = u(:, tel + (1:size(A{kk}, 2)));
            tel = tel + size(A{kk}, 2);
        end
        
        % Store R for the current block
        R{k} = r;
    end

%%%% Singular Value Decomposition (SVD) Function
function [U, A] = polsvd_c(X)
    [u, s, v] = svd(X, 'econ');
    
    % Determine truncation threshold
    if s(1, 1) < 1e-12
        t = 0;
    else
        t = sum(diag(s) / s(1, 1) > 1e-12);
    end
    
    % Truncate u and v matrices if needed
    if t < size(s, 1)
        u(:, t+1:end) = zeros(size(u, 1), size(u, 2) - t);
        v(:, t+1:end) = zeros(size(v, 1), size(v, 2) - t);
    end
    
    % Compute output matrices
    U = u * v';
    A = v * s * v';


%%%% Vectorize Function
function a = vec(b)
    a = reshape(b, [numel(b), 1]);


%%%% Generate SVD Indices
function [indxAA, NindxAA, svd_ind, svd_ind_l, svd_ind_r] = svd_all_indices(chi_l, chi_r, mps_ind, mps2_i, nblock, ind_middle_is_j, NN)
    indxAA = cell(NN, 1);
    tel = 0;
    
    % Generate indices for AA
    for k = 1:NN
        ql = mps_ind{mps2_i{k}(1)}(1);
        qr = mps_ind{mps2_i{k}(2)}(4);
        indxAA{k} = tel + (1:chi_l(ql) * chi_r(qr))';
        tel = tel + chi_l(ql) * chi_r(qr);
    end
    NindxAA = tel;
    
    % Initialize SVD indices
    svd_ind = cell(nblock, 1);
    svd_ind_l = cell(nblock, 1);
    svd_ind_r = cell(nblock, 1);
    
    % Generate SVD indices for each block
    for k = 1:nblock
        q = ind_middle_is_j{k};
        qql = []; qqr = [];
        
        for kk = 1:numel(q)
            qq = mps2_i{q(kk)};
            qql = [qql, qq(1)];
            qqr = [qqr, qq(2)];
        end
        
        % Get unique indices
        qql = unique(qql);
        qqr = unique(qqr);
        
        % Calculate chi_l and chi_r indices
        chil = cell(max(qql), 1);
        chir = cell(max(qqr), 1);
        tel = 0;
        
        for kk = 1:numel(qql)
            ch = chi_l(mps_ind{qql(kk)}(1));
            chil{qql(kk)} = tel + (1:ch);
            tel = tel + ch;
        end
        
        tel = 0;
        for kk = 1:numel(qqr)
            ch = chi_r(mps_ind{qqr(kk)}(4));
            chir{qqr(kk)} = tel + (1:ch);
            tel = tel + ch;
        end
        
        % Assign left and right indices
        chilr = cell(numel(q), 2);
        for kk = 1:numel(q)
            chilr{kk, 1} = chil{mps2_i{q(kk)}(1)};
            chilr{kk, 2} = chir{mps2_i{q(kk)}(2)};
        end
        
        svd_ind{k} = chilr;
        svd_ind_l{k} = chil;
        svd_ind_r{k} = chir;
    end


%%%% Generate Indices for rho and A
function [indrho, indA] = indrho_indA(chil, chir, nblock, mps_indices)
    % Generate indices for rho
    indrho = cell(nblock, 1);
    tell = 0;
    for k = 1:nblock
        p = chil(k) * chir(k);
        indrho{k} = tell + (1:p);
        tell = tell + p;
    end
    
    % Generate indices for A
    indA = cell(numel(mps_indices), 1);
    tell = 0;
    for k = 1:numel(indA)
        q = mps_indices{k};
        p = chil(q(1)) * chir(q(4));
        indA{k} = tell + (1:p);
        tell = tell + p;
    end

