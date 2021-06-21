function collision = check_collision_multi(obs_centers, obs_radii, traj)

collision = 0;

num_obs = size(obs_centers,1);
T = size(traj,1);
x1 = traj(1:T-1, 1);
y1 = traj(1:T-1, 2);
x2 = traj(2:T, 1);
y2 = traj(2:T, 2);
upperx = max(x1,x2);
lowerx = min(x1,x2);
uppery = max(y1,y2);
lowery = min(y1,y2);

a = [x1, y1];
temp = [x2 - x1, y2 - y1];
n = temp ./ vecnorm(temp')';

swing_radius = 167.5;

for i=1:1:num_obs
    p = obs_centers(i,:);
    radius = obs_radii(i,:);

    perp = (a - p) - sum((a-p).*n, 2).*n;
    int = p + perp;
    dist = vecnorm(perp')' - radius - swing_radius;

    in_line_segment = lowerx <= int(:,1) & int(:,1) <= upperx & lowery <= int(:,2) & int(:,2) <= uppery;

    if max(in_line_segment & (dist < 0))
        collision = 1;
    end
end

