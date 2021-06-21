function traj = generate_trajectory(imu, prim_mean, prim_std)
n = length(prim_mean(:,1));
traj = zeros(n, length(imu));
max_std = max(prim_std);
traj_deviation = normrnd(zeros(1,6), max_std);

for i=1:6
    deviation = traj_deviation(i) * prim_std(:, i)/max_std(i);
    traj(:,i) = prim_mean(:,i) + deviation;
end

end
