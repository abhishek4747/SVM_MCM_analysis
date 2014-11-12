%% is_in: function description
function [inside] = is_in(tetra, point)
	n = size(tetra, 1);
	D0 = [tetra ones(n, 1)];
	D1 = D0;
	D1(1,:)=[point 1];
	D2 = D0;
	D2(2,:)=[point 1];
	D3 = D0;
	D3(3,:)=[point 1];

	v = sum([det(D0) det(D1) det(D2) det(D3)] >= 0);
	
	if(v==n || v==0 )
		inside = true;
	else
		inside = false;
	end

end
