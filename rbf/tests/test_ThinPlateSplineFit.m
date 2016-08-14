centers = [1, 2; 
    3, 4;
    5, 6;
    1, 1;
    2, 2;
    3, 3;
    4, 4;
    5, 5];

% Returns ||x_i -x_j|| for x_i,x_j in centers
distance_matrix = DMatrix(centers,centers);

% r^2 log(r)  
tps = distance_matrix.*distance_matrix.*log(distance_matrix);
tps(isnan(tps))=0;

num_centers = length(centers);

% Generates linear polynomial matrix
P = [centers ones(num_centers,1)];

% Interpolation Matrix
M = [tps P; P' zeros(3,3)];

% sin(2*pi*x) * sin(2*pi*y)
data_values = sin(pi/7*centers(:,1)).*sin(pi/7*centers(:,2));

fit_solution = M\[data_values; zeros(3,1)];

interp_answer = M*fit_solution;
fit_error = interp_answer(1:end-3) - data_values;
disp(['The fit error is', num2str(norm(fit_error))])
