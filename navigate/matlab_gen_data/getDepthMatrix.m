% The problem is solved by identifying the (x,y) coordinates where the 3D
% line's projection on the horizontal plane intersects the cylinder,
% Once we have the (x,y) coordinates of the intersection, we can
% simply plug them in the equaltion of the ray to get z.
% Source: https://math.stackexchange.com/a/2613826

function deptharray = getDepthMatrix(imu, ths, phis, obs, walls, maxrng)
% imu = getIMUdata(V, swing);

% ths = zeros(1, length(ths_cell));
% for i = 1:length(ths_cell)  % converting this into an array instead of cell array
%     ths(i) = ths_cell{i};
% end

ths = mod(ths,360);
x0 = imu(1);
y0 = imu(2);
heading = imu(6) + 180; %may want to update this value to be related to previous frames?...
deptharray = double(maxrng)*ones([length(phis),length(ths)]);

nobs = length(obs(:,1));
horizontal_angle = mod(ths + heading,360); %viewing angle for this depth sensor
vertical_angle = mod(phis, 360);
% Slope for projection of the ray on the horizontal plane
m = tan((90+horizontal_angle)*pi/180);

for ob = 1:nobs
    %(x-a)^2 + (y-b)^2 = r^2
    a = obs(ob,1);
    b = obs(ob,2);
    r = obs(ob,3);
    % for i = 1:length(phis)


    d = y0 - m*x0;
    %begin check for intersection between like and obs
    del = max(r.^2*(1 + m.^2) - (b - m*a - d).^2,0);

    x1 = (a + b.*m - d.*m + sqrt(del)) ./ (1+m.^2);
    y1 = (d + a.*m + b.*m.^2 + m.*sqrt(del)) ./ (1+m.^2);
    x2 = (a + b.*m - d.*m - sqrt(del)) ./ (1+m.^2);
    y2 = (d + a.*m + b.*m.^2 - m.*sqrt(del)) ./ (1+m.^2);
    centx = x1 - x0;
    centy = y1 - y0;

    con1 = (270 <= horizontal_angle | horizontal_angle < 90) & centy >= 0;
    con2 = (90 <= horizontal_angle & horizontal_angle < 270) & centy <= 0;
    con3 = (0 <= horizontal_angle & horizontal_angle < 180) & centx <= 0;
    con4 = (180 <= horizontal_angle) & centx >= 0;

    ints = sum([con1; con2; con3; con4]) == 2 & del > 0;
    for j = find(ints == 1)
        dist1 = sqrt((x0-x1(j)).^2 + (y0-y1(j)).^2);
        dist2 = sqrt((x0-x2(j)).^2 + (y0-y2(j)).^2);
        % Project dist to incorporate vertical translation as well
        dist1 = dist1./cosd(vertical_angle);
        dist2 = dist2./cosd(vertical_angle);
        deptharray(:,j) = min([deptharray(:,j)'; dist1; dist2])';
    end

end

%walls
nwalls = length(walls);
for wall = 1:nwalls
    for i = 1:length(phis)
        for j = 1:length(ths)
            horizontal_angle = mod(ths(j) + heading,360); %viewing angle for this depth sensor
            vertical_angle = mod(phis(i), 360);
            m = tan(deg2rad(90+horizontal_angle));
            d = y0 - m*x0;
            %y = mx + b
            if (walls{wall}{1}=='y')
                y1 = double(walls{wall}{2});
                x1 = (y1-d)/m;
            elseif (walls{wall}{1}=='x')
                x1 = double(walls{wall}{2});
                y1 = m*x1 + d;
            end
            centx = x1 - x0;
            centy = y1 - y0;
            con1 = (270 <= horizontal_angle || horizontal_angle < 90) && centy >= 0;
            con2 = (90 <= horizontal_angle && horizontal_angle < 270) && centy <= 0;
            con3 = (0 <= horizontal_angle && horizontal_angle < 180) && centx <= 0;
            con4 = (180 <= horizontal_angle) && centx >= 0;
            if (sum([con1, con2, con3, con4]) == 2)
                dist = sqrt((x0-x1)^2 + (y0-y1)^2);
                % Project dist to incorporate vertical translation as well
                dist = dist/cosd(vertical_angle);
                deptharray(i,j) = min([deptharray(i,j), dist]);
            end
        end
    end
end


end
