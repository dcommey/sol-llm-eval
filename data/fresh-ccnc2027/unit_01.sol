pragma solidity ^0.8.27;
contract Unit {
    mapping(address => uint256) public credit;
    function deposit() external payable { credit[msg.sender] += msg.value; }
    function withdraw() external {
        uint256 amount = credit[msg.sender]; require(amount > 0);
        (bool ok,) = msg.sender.call{value: amount}(""); require(ok);credit[msg.sender] = 0;
    }
    
}
