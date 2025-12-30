function [P, A, Z,S, M, E] = TLF_MVC(X, n, k, c, param)
% 
num_view = length(X);
for i = 1:num_view
    A{i} = zeros(size(X{i},1), c*k);
    S{i} = zeros(c*k,n); 
    M{i} = zeros(c*k,n);
    E{i} = zeros(c*k,n);
    Z{i} = zeros(c*k,n);
     % Lagrange multipliers
    Q1{i} = zeros(c*k,n);
    Q2{i} = zeros(c*k,n); 
    Q3{i} = zeros(c*k,n);
    U{i} = zeros(c*k,n);
    K{i} = zeros(c*k,n);
end

lambda1 = param.lambda1;
lambda2 = param.lambda2;
lambda3 = param.lambda3;

if size(X{1},2)~=n
    for iv = 1:num_view
        X{iv} = X{iv}';
    end
end
for iv = 1:num_view
    X{iv} = NormalizeFea(X{iv},0);
end
clear iv

% initialization
I = eye(k); Y= [];
for ik = 1:k
    Y = [Y repmat(I(:,ik),1,c)];
end
clear I


for iv = 1:num_view
    [Up,~,Sp] = svd(A{iv}*Y', 'econ');
    W{iv} = Up*Sp';
end
clear Up Sp

P = zeros(n,c*k);
flag = 1;
MAX_ITER = 20;
rho = 0.01; 
mu = 2;
iter = 1;
while flag
iter = iter +1;


 P_temp = P;
       %% 1. Z step
    
    for iv = 1:num_view
        tZ1 = A{iv}'*X{iv};
        tZ2 = S{iv}+M{iv}+E{iv}-(Q1{iv}/rho);
        Z{i} = (rho*eye(c*k)/2 + A{i}'*A{i})^-1 * (tZ1+rho*tZ2/2);
    end
    clear tZ1 tZ2

        %   2.  A step


    for iv = 1:num_view
        
        A{iv} = (X{iv}*Z{iv}' + lambda1*W{iv}*Y)/(Z{i}*Z{i}' + lambda1*eye(c*k));
    end
    clear Up Sp

    %% 3.  W step
     for iv = 1:num_view
        [Up,~,Sp] = svd(A{iv}*Y', 'econ');
        W{iv} = Up*Sp';
     end
     clear Up Sp


        %% 4.  U step
    C = [];
    for i = 1:num_view
        slice = E{i}+Q3{i}/rho;
        C = [C; slice(:,:)];
    end
    [Wconcat] = solve_l1l2(C, 1/rho);
    for i = 1:num_view
        U{i} = Wconcat((i-1)*c*k+1:i*c*k,:);
    end

     %% 5.  K step
    S_tensor = cat(3, S{:,:});
    Q2_tensor = cat(3, Q2{:,:});
    s1 = S_tensor(:);
    q2 = Q2_tensor(:);
    [j, ~] = wshrinkObj(s1 + (1/rho)*q2,lambda2/rho,[c*k,n,num_view],0,1);
    K_tensor = reshape(j, [c*k,n,num_view]);
    for i = 1:num_view
        K{i} = K_tensor(:,:,i);
    end
   
    
    %%6.   S step
    
     M_tensor = cat(3, M{:,:}); % construct tensor from S2{i
    Mhat_tensor = fft(M_tensor,[],3); 
    E_tensor = cat(3, E{:,:}); % construct tensor from E{i}
    Ehat_tensor = fft(E_tensor,[],3);
    Q1_tensor = cat(3, Q1{:,:}); % construct tensor from Q1{i}
    Q1hat_tensor = fft(Q1_tensor,[],3);
    Z_tensor = cat(3, Z{:,:}); % construct tensor from Z{i}
    Zhat_tensor = fft(Z_tensor,[],3);
    K_tensor = cat(3, K{:,:}); % construct tensor from K{i}
   Khat_tensor = fft(K_tensor,[],3);
    Q2_tensor = cat(3, Q2{:,:}); % construct tensor from Q2{i}
    Q2hat_tensor = fft(Q2_tensor,[],3);
    for i = 1:num_view
        Shat_tensor(:,:,i) = (Khat_tensor(:,:,i)+Zhat_tensor(:,:,i)-Ehat_tensor(:,:,i)-Mhat_tensor(:,:,i)+Q1hat_tensor(:,:,i)/rho-Q2hat_tensor(:,:,i)/rho)/2;
    end
    S_tensor = ifft(Shat_tensor,[],3);
    for i = 1:num_view
        S{i} = S_tensor(:,:,i);
    end

    %%7.   M step
    S_tensor = cat(3, S{:,:}); % construct tensor from A{i}
   Shat_tensor = fft(S_tensor,[],3); 
    Q1_tensor = cat(3, Q1{:,:}); % construct tensor from Q1{i}
    Q1hat_tensor = fft(Q1_tensor,[],3);
    Z_tensor = cat(3, Z{:,:}); % construct tensor from S1{i}
   Zhat_tensor = fft(Z_tensor,[],3);
    E_tensor = cat(3, E{:,:}); % construct tensor from E{i}
   Ehat_tensor = fft(E_tensor,[],3);
    for i = 1:num_view
        Mhat_tensor(:,:,i) = -rho*(Shat_tensor(:,:,i)+Ehat_tensor(:,:,i)-Zhat_tensor(:,:,i)-Q1hat_tensor(:,:,i)/rho)/(2*lambda3+rho);
    end
    M_tensor = ifft(Mhat_tensor,[],3);
    for i = 1:num_view
        M{i} = M_tensor(:,:,i);
    end

    %%8.   E step
    S_tensor = cat(3, S{:,:}); % construct tensor from A{i}
   Shat_tensor = fft(S_tensor,[],3); 
     M_tensor = cat(3, M{:,:}); % construct tensor from S2{i}
    Mhat_tensor = fft(M_tensor,[],3); 
    Q1_tensor = cat(3, Q1{:,:}); % construct tensor from Q1{i}
    Q1hat_tensor = fft(Q1_tensor,[],3);
    Z_tensor = cat(3, Z{:,:}); % construct tensor from Z{i}
    Zhat_tensor = fft(Z_tensor,[],3);
    U_tensor = cat(3, U{:,:}); % construct tensor from K{i}
   Uhat_tensor = fft(U_tensor,[],3);
    Q3_tensor = cat(3, Q3{:,:}); % construct tensor from Q2{i}
    Q3hat_tensor = fft(Q3_tensor,[],3);
    for i = 1:num_view
        Ehat_tensor(:,:,i) = (Zhat_tensor(:,:,i)+Uhat_tensor(:,:,i)-Shat_tensor(:,:,i)-Mhat_tensor(:,:,i)+Q1hat_tensor(:,:,i)/rho-Q3hat_tensor(:,:,i)/rho)/2;
    end
    E_tensor = ifft(Ehat_tensor,[],3);
    for i = 1:num_view
        E{i} = E_tensor(:,:,i);
    end

      %%  Q1{i}, Q2, Q3{i} step
    for i = 1:num_view
        Q1{i} = Q1{i} + rho*(Z{i}-(S{i}+M{i})-E{i});
        Q2{i} = Q2{i} + rho*(S{i}-K{i});
        Q3{i} = Q3{i} + rho*(E{i}-U{i});
    end
    
     %%  rho step
    rho = rho*mu;


    tempS1=0;tempS2=0;
    for i = 1:num_view
        tempS1 = tempS1 + S{i};
        tempS2 = tempS2 + M{i};
    end
    P = (tempS2'+tempS1')/num_view;
     

   converge(iter)=norm(P-P_temp,'fro')^ 2/norm(P,'fro')^ 2;

    if (iter>0) && (converge(iter)<1e-5 || iter>MAX_ITER )
  
        flag = 0;
    end
end


 



