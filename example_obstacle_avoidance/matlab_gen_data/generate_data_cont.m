num_prims = length(prims);
img_size = 50;
T = length(prims{1}{1});

envs = zeros(num_envs, num_obs, 3);
% depth_maps = zeros(num_envs, num_prims, T, img_size, img_size);
depth_maps = zeros(num_envs, img_size, img_size);
min_dists = zeros(num_envs, num_prims);
env_prim_collision = zeros(num_envs, num_prims);
trajectories = zeros(num_envs, num_prims, T, 6);


% Note: all distances are in mm
imu = zeros(1,6) + [0 0 1000 0 0 -180];
num_rays = img_size;
ths = linspace(-60, 60, num_rays);
phis = linspace(-60, 60, num_rays);
maxrng = 7000;  % range of depth sensor in mm
stdev_scale = 1.5;

x_lim = [-3500, 3500];
y_lim = [4000, maxrng];


num_obs = num_obs + round(variable_difficulty);
not_vd = variable_difficulty == 0;
vd = variable_difficulty;

cond_ind = [easy, hard, ~not_vd, ir_shift];


for i=1:1:num_envs
    if mod(i, 100) == 0
        disp(i)
    end
    
    while true
        if num_obs > 0
        obs_r = 202.9*ones(num_obs,1);
        obs_x_lim = [x_lim(1) + obs_r(1), x_lim(2) - obs_r(1)];
        obs_y_lim = [y_lim(1) + obs_r(1), y_lim(2) - obs_r(1)];
        obs_x = (obs_x_lim(2)-obs_x_lim(1))*rand(num_obs,1) + obs_x_lim(1)*ones(num_obs,1);
        obs_y = (obs_y_lim(2)-obs_y_lim(1))*rand(num_obs,1) + obs_y_lim(1)*ones(num_obs,1);
        obs = [obs_x, obs_y, obs_r];
        else
            obs = [-1, -1, -1];
        end
        
        for j=1:num_prims
            k = randsample(length(prims{j}),1);
            traj = prims{j}{k};
            
            trajectories(i, j, :, :) = traj(1:T,:);
            
            min_dist = get_min_dist(obs, traj);
            min_dists(i,j) = min_dist;
        end
        
        exit_cond(1) = max(min_dists(i,:))>100 && min(cond_ind == [0,0,0,0]);
        exit_cond(2) = max(min_dists(i,:))>300 && min(cond_ind == [1,0,0,0]);
        exit_cond(3) = max(min_dists(i,:))>0 && max(min_dists(i,:))<300 && min(cond_ind == [0,1,0,0]);
        exit_cond(4) = max(min_dists(i,:))>100-(10 * vd) && min(cond_ind == [1,0,1,0]);
        exit_cond(5) = max(min_dists(i,:))>100-(12 * vd) && max(min_dists(i,:))<100*max(20 - sign(vd)*vd*vd,1) && min(cond_ind == [0,0,1,0]);
        exit_cond(6) = max(min_dists(i,:))>100 && max(min_dists(i,:))<550 && min(cond_ind == [0,0,0,1]);
        if max(exit_cond)
            deptharray = getDepthMatrix(imu, ths, phis, obs, {}, maxrng);
            deptharray = deptharray/maxrng;
            deptharray = flip(deptharray,2);
            depth_maps(i,:,:) = deptharray;

            break
            

        end
        

    end
    
    if verbose
        figure(1);
        hold on
        viscircles(obs(:,1:2), obs(:,3));
        min_dists(1,:);
        softmax(min_dists(1,:)'/1e3)
        daspect([1 1 1]);
        for j = 1:num_prims
            scatter(trajectories(1,j,:,1),trajectories(1,j,:,2),'o')
        end

        break
    end
end

solvable_envs = sum(sum(min_dists>0,2) > 0);
prim_collision = double(min_dists<=0);

norm_min_dist = max(min(min_dists/300, 1), 0);
prim_cost = 1 - norm_min_dist;
prim_cost(1,:);

dist_softmax = softmax(min_dists'/1e3)';



loc = folder;
app = add_app + ".m";


if ~verbose
    save(loc + 'depth_maps' + app,'depth_maps');
    save(loc + 'dist_softmax' + app ,'dist_softmax');
    save(loc + 'prim_cost' + app,'prim_cost');  % continuous cost based on distance to closest obstacle
    % save(loc + 'prim_collision' + app,'prim_collision');  % boolean on if collision happened
end
