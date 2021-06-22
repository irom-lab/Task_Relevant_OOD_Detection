function min_dist = get_min_dist_depth(depth, ths, phis, maxrng)
% using depth sensor to determine depth
% simple heuristic based on the fact that the swing does not turn very
% aggresively. We can approximate how close obstacles will get if there is
% a fly-by.

[C, I] = min(depth);  % vertical
[C2, I2] = min(C);    % horizontal
r = 167.5;  % approximate radius of swing
min_dist = C2*maxrng;
phi = phis(I(I2));
th = abs(ths(I2));
if th>45
    min_dist = min_dist * cosd(90 - th);
end

min_dist = min_dist - r;

end

