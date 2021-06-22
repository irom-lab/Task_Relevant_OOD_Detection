function min_dist = get_min_dist(obs, prim_mean)
% minimum distance of the obstacles using full state knowledge

if obs ==[-1, -1, -1]
    min_dist = 1e8;
else
    obs_centers = obs(:,1:2);
    obs_radii = obs(:,3);

    num_obs = size(obs_centers,1);
    min_dist = 1e10;

    r = 167.5;  % approximate radius of swing

    for i=1:num_obs
        center = obs_centers(i,:);
        radius = obs_radii(i,:);

        delx = prim_mean(:,1) - center(1);
        dely = prim_mean(:,2) - center(2);
        % dist = vecnorm(center' - prim_mean(:,1:2)') - r - radius;
        dist = sqrt(delx.^2 + dely.^2) - r - radius;

        min_dist = min(min(dist),min_dist);

    end    
end
end

