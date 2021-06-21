function min_dist = check_collision(obs_centers, obs_radii, prim_mean, prim_std)
%CHECK_COLLISION Returns minimum distance of the obstacles from a primitive
% trajectory distribution
num_obs = size(obs_centers,1);
min_dist = 1e10;

% Overapproximate the ellipse of uncertainty for X,Y by a circle by
% taking the max std of X and Y and using that as the radius
a = prim_std(:,1);
b = prim_std(:,2);
r = max([a,b]'); %#ok<UDIM>

for i=1:num_obs
    center = obs_centers(i,:);
    radius = obs_radii(i,:);

    % Compute distance of the center of the uncertainty circle from the
    % center of the obstacle circle and subtract their radii to
    % identify if the trajectory will contact the obstacle. If dist is
    % negative, then collision occurs.
    dist = vecnorm(center' - prim_mean(:,1:2)') - r - radius;


    if min(dist) <= min_dist
        min_dist = min(dist);
    end
end    

end

