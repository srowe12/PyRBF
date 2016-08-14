function DM=DMatrix(p1,p2)


%np1 = [p1(:,1).^2+p1(:,2).^2]*ones(1,length(p2));
%np2 = [p2(:,1).^2+p2(:,2).^2]*ones(1,length(p1));
np1 = sum(p1.*p1,2);
np2 = sum(p2.*p2,2);
DM = sqrt(np1(:,ones(1,size(p2,1)))+np2(:,ones(1,size(p1,1)))'-2*p1*p2');
%DM = sqrt(np1-2*p1*p2' +np2');


