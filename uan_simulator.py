# -*- Mode:Python; -*-
# /*
#  * Copyright (c) 2022 Hanoi University of Science and Technology
#  *
#  * This program is free software; you can redistribute it and/or modify
#  * it under the terms of the GNU General Public License version 2 as
#  * published by the Free Software Foundation;
#  *
#  * This program is distributed in the hope that it will be useful,
#  * but WITHOUT ANY WARRANTY; without even the implied warranty of
#  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  * GNU General Public License for more details.
#  *
#  * You should have received a copy of the GNU General Public License
#  * along with this program; if not, write to the Free Software
#  * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#  *
#  * Authors: Chung Duc Nguyen Dang <nguyendangchungduc1999@gmail.com>
#  */

import ns.core 
import ns.uan
import ns.mobility
import ns.energy 
import ns.network
import ns.energy
import ns.internet
import numpy as np 
from reinforcement import Reinforcement
from pathPlanning import PathPlanning

def PrintReceivedPacket(socket):
    srcAddress = ns.network.Address()
    while (socket.GetRxAvailable() > 0):
        packet = socket.RecvFrom(srcAddress)
        packetSocketAddress = ns.network.PacketSocketAddress.ConvertFrom(srcAddress)
        srcAddress = packetSocketAddress.GetPhysicalAddress()
        packet.CopyData (message, 1)
        if (ns.network.Mac8Address.IsMatchingType(srcAddress)):
            print("Time: ", ns.core.Simulator.Now().GetHours(), "h", srcAddress, " message: ", message)

def SendSinglePacket(node, socket, pkt, dst):
    print("Time: ", ns.core.Simulator.Now().GetHours(), "h", " packet sent to node", dst, " from ", node.GetDevice(0))
    socketAddress = ns.network.PacketSocketAddress()
    socketAddress.SetSingleDevice(node.GetDevice(0).GetIfIndex())
    socketAddress.SetPhysicalAddress(dst)
    socketAddress.SetProtocol(0)
    socket.SendTo(pkt, 0, socketAddress)

if __name__ == '__main__':
    print("UAN")
    m_nodes = ns.network.NodeContainer()
    m_nodes.Create(2)
    mobilityHelper = ns.mobility.MobilityHelper()
    mobilityHelper.SetMobilityModel("ns3::ConstantPositionMobilityModel")
    mobilityHelper.Install(m_nodes)
    m_nodes.Get(0).GetObject(ns.mobility.MobilityModel.GetTypeId()).SetPosition(ns.core.Vector(0, 0, 0))
    m_nodes.Get(1).GetObject(ns.mobility.MobilityModel.GetTypeId()).SetPosition(ns.core.Vector(1000, 1000, 0))

    channel = ns.uan.UanChannel()
    uanHelper = ns.uan.UanHelper()
    netDeviceContainer = uanHelper.Install(m_nodes, channel)

    packetSocketHelper = ns.network.PacketSocketHelper()

    packetSocketHelper.Install(m_nodes.Get(0))
    socketAddress = ns.network.PacketSocketAddress()
    socketAddress.SetSingleDevice(m_nodes.Get(0).GetDevice(0).GetIfIndex ())
    socketAddress.SetProtocol(0)
    sockets = ns.network.Socket.CreateSocket(m_nodes.Get(0), ns.core.TypeId.LookupByName("ns3::PacketSocketFactory"))
    sockets.Bind()
    sockets.Connect(socketAddress)
    sockets.SetRecvCallback(PrintReceivedPacket)

    packetSocketHelper.Install(m_nodes.Get(1))
    socketAddress = ns.network.PacketSocketAddress()
    socketAddress.SetSingleDevice(m_nodes.Get(1).GetDevice(0).GetIfIndex ())
    socketAddress.SetProtocol(0)
    sockets1 = ns.network.Socket.CreateSocket(m_nodes.Get(1), ns.core.TypeId.LookupByName("ns3::PacketSocketFactory"))
    sockets1.Bind()
    sockets1.Connect(socketAddress)
    sockets1.SetRecvCallback(PrintReceivedPacket)

    uniformRandomVariable = ns.core.UniformRandomVariable()
    dst = ns.network.Mac8Address.ConvertFrom(m_nodes.Get(0).GetDevice(0).GetAddress())
    dst1 = ns.network.Mac8Address.ConvertFrom(m_nodes.Get(1).GetDevice(0).GetAddress())

    pkt = ns.network.Packet(10)
    
    packet = ns.network.Packet(1024)

    for i in range(4):
        time = uniformRandomVariable.GetValue(0, 60)
        ns.core.Simulator.Schedule(ns.core.Seconds(time), SendSinglePacket, m_nodes.Get(1), sockets, packet, dst)

        time = uniformRandomVariable.GetValue(0, 60)
        ns.core.Simulator.Schedule(ns.core.Seconds(time), SendSinglePacket, m_nodes.Get(0), sockets1, pkt, dst1)

        time1 = 240
    
    ns.core.Simulator.Schedule(ns.core.Seconds(time1), Reinforcement)
    ns.core.Simulator.Schedule(ns.core.Seconds(time1*2), PathPlanning)
    ns.core.Simulator.Stop(ns.core.Days(50))
    ns.core.Simulator.Run()
    ns.core.Simulator.Destroy()

